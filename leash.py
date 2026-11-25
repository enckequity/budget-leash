"""Budget Leash engine — hard spend caps for Hermes sessions.

Most agent "budgets" are dashboards: they tell you what you spent after the
money is gone. The Leash enforces *before* dispatch. It accumulates two
meters per session and per UTC day:

  * estimated model spend — token usage from ``post_api_request`` priced
    with a configurable per-model table (USD per 1M tokens), and
  * tool calls — counted at ``post_tool_call``, with optional flat per-tool
    costs for tools that bill outside the model (paid APIs, rentals).

Cache tokens are priced too: providers report cache reads/writes in
separate usage fields, and those tokens are charged on top of input/output
unless the price table says otherwise (see ``_usage_cost``).

``pre_tool_call`` compares the meters against a per-session cap and a
persistent per-day cap and returns a block decision when either is
exhausted. Tool-call blocking alone left a hole, though: a model that
keeps *talking* — reasoning turns that never touch a tool — still burns
spend after the cap. The ``llm_execution`` middleware closes it: once a
cap is exhausted the provider call itself is never made, and the agent
loop receives a synthetic "budget exhausted" reply instead (see
:func:`budget_stop_response`). The day ledger lives on disk and survives restarts, so an
unattended cron fleet shares one daily ceiling instead of one per process.
Every ledger read and write happens under an exclusive file lock
(``flock``) and re-reads the file first, so two processes (CLI + gateway)
can no longer overwrite each other's totals — last-writer-wins is gone.
At 80% of a cap the model gets a one-time context warning; at 100% tool
calls stop — the agent can still write its final answer explaining why,
because plain replies are not tool calls.

Fail-open on internal errors (the plugin layer swallows exceptions), but
fail-closed on budget: when the cap says no, the call does not run.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

DEFAULTS: dict[str, Any] = {
    "session_cap_usd": 2.00,
    "daily_cap_usd": 10.00,
    "session_tool_call_cap": 200,
    "daily_tool_call_cap": 1000,
    "warn_fraction": 0.80,
    "tool_costs": {},  # tool_name -> flat USD charged per completed call
    "prices": {},  # model -> {"input": usd_per_1m, "output": usd_per_1m}
    "default_price": {"input": 1.00, "output": 3.00},
    "disabled": False,
}

# List prices, USD per 1M tokens, standard tier, as of 2026-11-24. Sources: Anthropic API
# pricing (cache write = 5-minute write, 1.25x input) and OpenAI API pricing (short context,
# <=272K input; OpenAI does not charge for cache writes). Used when ``prices`` in
# budget-leash.json has no entry for the model; an explicit entry there always wins.
PRICES_AS_OF = "2026-11-24"


def _anthropic(pin: float, pout: float, pread: Optional[float] = None) -> dict[str, float]:
    return {"input": pin, "output": pout,
            "cache_read": pread if pread is not None else round(pin * 0.1, 4),
            "cache_write": round(pin * 1.25, 4)}


def _openai(pin: float, pout: float, pread: Optional[float] = None) -> dict[str, float]:
    return {"input": pin, "output": pout,
            "cache_read": pread if pread is not None else pin, "cache_write": pin}


BUILTIN_PRICES: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-fable-5-2": _anthropic(8.00, 40.00, 0.20),
    "claude-mythos-5-2": _anthropic(8.00, 40.00, 0.20),
    "claude-fable-5-1": _anthropic(10.00, 50.00),
    "claude-mythos-5-1": _anthropic(10.00, 50.00),
    "claude-fable-5": _anthropic(10.00, 50.00),
    "claude-opus-5-5": _anthropic(4.00, 20.00, 0.20),
    "claude-opus-5": _anthropic(5.00, 25.00),
    "claude-opus-4-8": _anthropic(5.00, 25.00),
    "claude-opus-4-7": _anthropic(5.00, 25.00),
    "claude-opus-4-6": _anthropic(5.00, 25.00),
    "claude-sonnet-5-5": _anthropic(2.00, 10.00, 0.10),
    "claude-sonnet-5": _anthropic(2.00, 10.00),
    "claude-sonnet-4-6": _anthropic(3.00, 15.00),
    "claude-haiku-5-5": _anthropic(0.10, 0.50),  # prompts up to 100K tokens
    "claude-haiku-4-5": _anthropic(1.00, 5.00),
    # OpenAI
    "gpt-6.2-astra": _openai(8.00, 40.00, 0.40),
    "gpt-6.2-sol": _openai(2.00, 10.00, 0.10),
    "gpt-6-astra": _openai(10.00, 50.00, 1.00),
    "gpt-6.1-sol": _openai(2.00, 10.00, 0.10),
    "gpt-6.1-terra": _openai(1.00, 5.00, 0.05),
    "gpt-6-luna": _openai(0.10, 0.50, 0.01),
    "gpt-6-sol": _openai(2.00, 10.00, 0.20),
    "gpt-5.6-sol": _openai(4.00, 20.00, 0.40),
    "gpt-5.6-terra": _openai(2.00, 12.00, 0.20),
    "gpt-5.6-luna": _openai(0.20, 1.20, 0.02),
    "gpt-5.5": _openai(5.00, 30.00, 0.50),
    "gpt-5.5-pro": _openai(30.00, 180.00),
    "gpt-5.4": _openai(2.50, 15.00, 0.25),
    "gpt-5.4-pro": _openai(30.00, 180.00),
    "gpt-5.4-mini": _openai(0.75, 4.50, 0.075),
    "gpt-5.4-nano": _openai(0.20, 1.25, 0.02),
    "gpt-5.3-codex": _openai(1.75, 14.00, 0.175),
    "gpt-5.2": _openai(1.75, 14.00, 0.175),
    "gpt-5.2-pro": _openai(21.00, 168.00),
    "gpt-5.1": _openai(1.25, 10.00, 0.125),
    "gpt-5": _openai(1.25, 10.00, 0.125),
    "gpt-5-mini": _openai(0.25, 2.00, 0.025),
    "gpt-5-nano": _openai(0.05, 0.40, 0.005),
    "gpt-5-pro": _openai(15.00, 120.00),
    "gpt-4.1": _openai(2.00, 8.00, 0.50),
    "gpt-4.1-mini": _openai(0.40, 1.60, 0.10),
    "gpt-4o": _openai(2.50, 10.00, 1.25),
    "gpt-4o-mini": _openai(0.15, 0.60, 0.075),
    "o3": _openai(2.00, 8.00, 0.50),
    "o3-pro": _openai(20.00, 80.00),
}

# Bare family aliases some providers pass as the model name.
_ALIASES = {
    "opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5",
    "haiku": "claude-haiku-5-5", "fable": "claude-fable-5-2",
}


def builtin_price(model: Optional[str]) -> Optional[dict[str, float]]:
    """The built-in list price for a model name as providers spell it, or None.

    Accepts a provider prefix (``anthropic/claude-sonnet-5.5``), dots or dashes in Claude
    version numbers, a trailing date snapshot (``-20260401``) and the bare family aliases.
    """
    name = (model or "").strip().lower()
    if not name:
        return None
    name = name.rsplit("/", 1)[-1].split(":", 1)[0]
    name = _ALIASES.get(name, name)
    if name.startswith("claude-"):
        name = name.replace(".", "-")
    parts = name.split("-")
    if len(parts) > 1 and parts[-1].isdigit() and len(parts[-1]) == 8:
        name = "-".join(parts[:-1])
    entry = BUILTIN_PRICES.get(name)
    if entry is None and name.startswith("claude-"):
        # claude-sonnet-5-5-latest and similar: longest known prefix.
        for key in sorted(BUILTIN_PRICES, key=len, reverse=True):
            if key.startswith("claude-") and name.startswith(key + "-"):
                entry = BUILTIN_PRICES[key]
                break
    return dict(entry) if entry else None


_ENV_MAP = {
    "LEASH_SESSION_CAP_USD": ("session_cap_usd", float),
    "LEASH_DAILY_CAP_USD": ("daily_cap_usd", float),
    "LEASH_SESSION_TOOL_CALLS": ("session_tool_call_cap", int),
    "LEASH_DAILY_TOOL_CALLS": ("daily_tool_call_cap", int),
}


def hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


def data_dir() -> Path:
    """Per-profile plugin data folder (reviewer recommendation): the ledger
    and the session reports live here instead of the Hermes home root, so in
    a multi-profile process each profile gets its own ledger."""
    return hermes_home() / "plugin-data" / "budget-leash"


def ledger_path() -> Path:
    return data_dir() / "ledger.json"


def reports_path() -> Path:
    return data_dir() / "reports.jsonl"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_config() -> dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    try:
        data = json.loads((hermes_home() / "budget-leash.json").read_text())
        if isinstance(data, dict):
            for key in cfg:
                if key in data:
                    cfg[key] = data[key]
    except Exception:
        pass
    for env, (key, cast) in _ENV_MAP.items():
        raw = os.environ.get(env)
        if raw:
            try:
                cfg[key] = cast(raw)
            except ValueError:
                pass
    if os.environ.get("LEASH_DISABLE", "").lower() in {"1", "true", "yes", "on"}:
        cfg["disabled"] = True
    return cfg


class Ledger:
    """Persistent per-day totals, shared across processes via the file.

    Fix for the daily-ledger race (reviewer finding 1): the old code loaded
    the file once at init and saved with last-writer-wins, so a CLI process
    and the gateway each counted their own spend and overwrote the other —
    the daily cap could multiply by the number of processes. Every mutation
    now goes through :meth:`apply`, which takes an exclusive ``flock`` on a
    sidecar lock file, re-reads the file, applies the delta, and writes it
    back before releasing. Every read path (:meth:`sync`, used by the
    pre-call checks and reports) re-reads under the same lock, so a
    long-lived process never acts on a stale total. The tmp file carries the
    pid so two processes never share one either.

    Counters only ever grow within a day (a new UTC day resets them), so
    :meth:`save` merges by taking the per-counter max for the same day
    instead of blindly overwriting — it is kept for compatibility only;
    internal code uses :meth:`apply`.

    If locking is unavailable (non-POSIX, or the lock file cannot be
    created), the ledger degrades to the old in-memory behaviour rather
    than raising: a Leash bug must never break the agent loop.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or ledger_path()
        if path is None:
            self._migrate_legacy_ledger()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.day = _today()
        self.spent_usd = 0.0
        self.tool_calls = 0
        self.api_calls = 0
        self._load()

    def _migrate_legacy_ledger(self) -> None:
        """One-time copy of the pre-0.1.2 ledger from the Hermes home root,
        so moving the file never resets a day's totals."""
        try:
            legacy = hermes_home() / "budget-leash-ledger.json"
            if legacy.exists() and not self.path.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(legacy, self.path)
        except Exception:
            pass

    # -- file access (caller must hold the lock, except _load) ---------------

    def _read_file(self) -> None:
        """Load state from disk; resets counters when the day rolled over."""
        try:
            data = json.loads(self.path.read_text())
        except Exception:
            return
        if not isinstance(data, dict):
            return
        if data.get("day") == self.day:
            self.spent_usd = float(data.get("spent_usd", 0.0))
            self.tool_calls = int(data.get("tool_calls", 0))
            self.api_calls = int(data.get("api_calls", 0))
        elif isinstance(data.get("day"), str) and data["day"] > self.day:
            # Another process already rolled to a newer day: adopt it.
            self.day = data["day"]
            self.spent_usd = float(data.get("spent_usd", 0.0))
            self.tool_calls = int(data.get("tool_calls", 0))
            self.api_calls = int(data.get("api_calls", 0))
        else:
            self.spent_usd = 0.0
            self.tool_calls = 0
            self.api_calls = 0

    def _write_file(self) -> None:
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({
            "day": self.day,
            "spent_usd": round(self.spent_usd, 6),
            "tool_calls": self.tool_calls,
            "api_calls": self.api_calls,
        }))
        tmp.replace(self.path)

    def _acquire(self):
        """Take the exclusive lock; return its fd, or None if unavailable."""
        if fcntl is None:
            return None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            return fd
        except Exception:
            return None

    @staticmethod
    def _release(fd) -> None:
        if fd is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(fd)
        except Exception:
            pass

    # -- public API ------------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
            if data.get("day") == self.day:
                self.spent_usd = float(data.get("spent_usd", 0.0))
                self.tool_calls = int(data.get("tool_calls", 0))
                self.api_calls = int(data.get("api_calls", 0))
        except Exception:
            pass

    def sync(self) -> None:
        """Re-read the on-disk totals under the lock (read paths use this)."""
        fd = self._acquire()
        try:
            if fd is not None:
                self.day = _today()
                self._read_file()
        except Exception:
            pass
        finally:
            self._release(fd)

    def apply(self, spent_usd: float = 0.0, tool_calls: int = 0,
              api_calls: int = 0) -> None:
        """Atomically add deltas to the shared daily totals.

        Re-reads the file under the lock first, so a delta computed by this
        process is added to the other processes' totals instead of replacing
        them. Handles the UTC day rollover the same way.
        """
        fd = self._acquire()
        try:
            today = _today()
            if today != self.day:
                self.day = today
                self.spent_usd = 0.0
                self.tool_calls = 0
                self.api_calls = 0
            if fd is not None:
                self._read_file()
                self.spent_usd += spent_usd
                self.tool_calls += tool_calls
                self.api_calls += api_calls
                self._write_file()
            else:
                self.spent_usd += spent_usd
                self.tool_calls += tool_calls
                self.api_calls += api_calls
                try:
                    self._write_file()
                except Exception:
                    pass
        except Exception:
            # Fail-open on ledger errors: keep the in-memory totals.
            self.spent_usd += spent_usd
            self.tool_calls += tool_calls
            self.api_calls += api_calls
        finally:
            self._release(fd)

    def roll(self) -> None:
        today = _today()
        if today != self.day:
            self.day = today
            self.spent_usd = 0.0
            self.tool_calls = 0
            self.api_calls = 0
            self.save()

    def save(self) -> None:
        """Compatibility save: merges in-memory totals with the file's.

        Takes the per-counter max for the same day (counters only grow
        within a day), so a stale in-memory copy can no longer clobber
        another process's totals. Prefer :meth:`apply` for new code.
        """
        fd = self._acquire()
        try:
            if fd is not None:
                try:
                    disk = json.loads(self.path.read_text())
                except Exception:
                    disk = {}
                if isinstance(disk, dict) and disk.get("day") == self.day:
                    self.spent_usd = max(self.spent_usd, float(disk.get("spent_usd", 0.0)))
                    self.tool_calls = max(self.tool_calls, int(disk.get("tool_calls", 0)))
                    self.api_calls = max(self.api_calls, int(disk.get("api_calls", 0)))
                self._write_file()
            else:
                try:
                    self._write_file()
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._release(fd)


@dataclass
class SessionMeter:
    session_id: str
    spent_usd: float = 0.0
    tool_calls: int = 0
    api_calls: int = 0
    blocked_calls: int = 0
    blocked_api_calls: int = 0
    warned: bool = False
    started_at: float = field(default_factory=time.time)


class Leash:
    def __init__(self, config: Optional[dict[str, Any]] = None, ledger: Optional[Ledger] = None) -> None:
        self.config = config if config is not None else load_config()
        self.ledger = ledger if ledger is not None else Ledger()
        self._sessions: dict[str, SessionMeter] = {}
        self._lock = threading.Lock()

    # -- helpers -------------------------------------------------------------

    def _meter(self, session_id: Optional[str]) -> SessionMeter:
        key = session_id or "default"
        m = self._sessions.get(key)
        if m is None:
            m = SessionMeter(session_id=key)
            self._sessions[key] = m
        return m

    def _price_for(self, model: Optional[str]) -> tuple[float, float, float, float]:
        """(input, output, cache_read, cache_write) USD per 1M tokens.

        Fix for the unpriced-cache-tokens finding (reviewer finding 3):
        providers report cache reads/writes in separate usage fields, so a
        price table that only covers input/output under-counts spend on
        cached providers. Per-model entries may set ``cache_read`` /
        ``cache_write`` explicitly (and so may ``default_price``); otherwise
        each defaults to that model's input price — a conservative
        over-estimate rather than a silent under-count. Configure the real
        rates (e.g. Anthropic cache read ~10% of input) for exactness.
        """
        prices = self.config.get("prices", {})
        entry = prices.get(model or "") or builtin_price(model) or self.config["default_price"]
        default = self.config["default_price"]
        pin = float(entry.get("input", 0.0))
        pout = float(entry.get("output", 0.0))
        pread = float(entry.get("cache_read", default.get("cache_read", pin)))
        pwrite = float(entry.get("cache_write", default.get("cache_write", pin)))
        return pin, pout, pread, pwrite

    @staticmethod
    def _usage_tokens(usage: Any) -> tuple[int, int, int, int]:
        """(input, output, cache_read, cache_write) token counts from a usage payload.

        Providers name the cache fields differently; cover the common ones:
        Anthropic (cache_read_input_tokens / cache_creation_input_tokens),
        OpenAI (prompt_tokens_details.cached_tokens), Gemini
        (cached_content_token_count), plus generic cache_read/cache_write
        spellings. Missing fields count as 0.
        """
        if not isinstance(usage, dict):
            return 0, 0, 0, 0
        in_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        details = usage.get("prompt_tokens_details")
        if not isinstance(details, dict):
            details = {}
        read_keys = ("cache_read_input_tokens", "cache_read_tokens",
                     "cached_tokens", "cached_content_token_count", "cache_read")
        write_keys = ("cache_creation_input_tokens", "cache_creation_tokens",
                      "cache_write_input_tokens", "cache_write_tokens", "cache_write")
        read_tok = 0
        for key in read_keys:
            read_tok = int(usage.get(key) or details.get(key) or 0)
            if read_tok:
                break
        write_tok = 0
        for key in write_keys:
            write_tok = int(usage.get(key) or details.get(key) or 0)
            if write_tok:
                break
        return in_tok, out_tok, read_tok, write_tok

    def _usage_cost(self, model: Optional[str], usage: Any) -> float:
        in_tok, out_tok, read_tok, write_tok = self._usage_tokens(usage)
        pin, pout, pread, pwrite = self._price_for(model)
        return (in_tok / 1e6 * pin + out_tok / 1e6 * pout
                + read_tok / 1e6 * pread + write_tok / 1e6 * pwrite)

    def _cap_reason(self, m: SessionMeter) -> str:
        cfg = self.config
        if m.spent_usd >= float(cfg["session_cap_usd"]):
            return f"session spend ${m.spent_usd:.2f} reached the session cap ${float(cfg['session_cap_usd']):.2f}"
        if self.ledger.spent_usd >= float(cfg["daily_cap_usd"]):
            return f"today's spend ${self.ledger.spent_usd:.2f} reached the daily cap ${float(cfg['daily_cap_usd']):.2f}"
        if m.tool_calls >= int(cfg["session_tool_call_cap"]):
            return f"session tool calls {m.tool_calls} reached the session cap {int(cfg['session_tool_call_cap'])}"
        if self.ledger.tool_calls >= int(cfg["daily_tool_call_cap"]):
            return f"today's tool calls {self.ledger.tool_calls} reached the daily cap {int(cfg['daily_tool_call_cap'])}"
        return ""

    # -- hook events -----------------------------------------------------------

    def on_post_api_request(self, session_id: Optional[str], model: Optional[str], usage: Any) -> None:
        if self.config.get("disabled"):
            return
        with self._lock:
            m = self._meter(session_id)
            m.api_calls += 1
            cost = 0.0
            if isinstance(usage, dict):
                # Cache reads/writes are priced on top of input/output
                # (reviewer finding 3); see _usage_cost for field names.
                cost = self._usage_cost(model, usage)
                m.spent_usd += cost
            # Atomic cross-process ledger update (reviewer finding 1):
            # re-reads the file under flock, adds the delta, writes back.
            self.ledger.apply(spent_usd=cost, api_calls=1)

    def on_pre_tool_call(self, session_id: Optional[str], tool_name: str) -> Optional[dict[str, str]]:
        """Return a block decision when a cap is exhausted, else None."""
        if self.config.get("disabled"):
            return None
        with self._lock:
            # Refresh the shared daily totals before deciding: another
            # process may have spent since this one loaded the ledger.
            self.ledger.sync()
            m = self._meter(session_id)
            reason = self._cap_reason(m)
            if not reason:
                flat = float(self.config.get("tool_costs", {}).get(tool_name, 0.0))
                if flat > 0 and m.spent_usd + flat > float(self.config["session_cap_usd"]):
                    reason = (f"this {tool_name} call costs ${flat:.2f} and would push session spend "
                              f"${m.spent_usd:.2f} over the session cap ${float(self.config['session_cap_usd']):.2f}")
            if reason:
                m.blocked_calls += 1
                return {
                    "action": "block",
                    "message": (f"[BUDGET LEASH] Tool call blocked: {reason}. "
                                "Stop calling tools and write your final answer explaining the budget stop. "
                                "The workspace owner can raise the budget caps."),
                }
            return None

    def on_post_tool_call(self, session_id: Optional[str], tool_name: str, status: Any) -> None:
        if self.config.get("disabled"):
            return
        with self._lock:
            if status == "blocked":
                return  # blocked calls never ran; don't meter them
            m = self._meter(session_id)
            m.tool_calls += 1
            flat = float(self.config.get("tool_costs", {}).get(tool_name, 0.0))
            if flat > 0:
                m.spent_usd += flat
            # Atomic cross-process ledger update (reviewer finding 1).
            self.ledger.apply(spent_usd=flat, tool_calls=1)

    def on_llm_execution(self, session_id: Optional[str]) -> Optional[str]:
        """Cut-off check for the ``llm_execution`` middleware (finding 2).

        Returns the cap reason when any cap is already exhausted — the
        plugin layer then answers the agent loop with a synthetic budget
        reply *without* invoking the provider, so a capped session cannot
        keep spending on tool-free reasoning turns. Counts each cut-off
        call on the session meter. Returns None while budget remains.
        """
        if self.config.get("disabled"):
            return None
        with self._lock:
            # Same freshness rule as the tool-call check: another process
            # may have spent since this one loaded the ledger.
            self.ledger.sync()
            m = self._meter(session_id)
            reason = self._cap_reason(m)
            if reason:
                m.blocked_api_calls += 1
                return reason
            return None

    def on_pre_llm_call(self, session_id: Optional[str]) -> Optional[str]:
        """One-time 80% warning so the model can wrap up before the hard stop."""
        if self.config.get("disabled"):
            return None
        with self._lock:
            m = self._meter(session_id)
            if m.warned:
                return None
            self.ledger.sync()  # warn on the shared daily total, not a stale copy
            frac = float(self.config["warn_fraction"])
            session_hit = m.spent_usd >= frac * float(self.config["session_cap_usd"])
            daily_hit = self.ledger.spent_usd >= frac * float(self.config["daily_cap_usd"])
            if session_hit or daily_hit:
                m.warned = True
                return (f"[BUDGET LEASH] Budget warning: session spend ${m.spent_usd:.2f} of "
                        f"${float(self.config['session_cap_usd']):.2f} cap; today ${self.ledger.spent_usd:.2f} of "
                        f"${float(self.config['daily_cap_usd']):.2f}. Tool calls will be blocked when a cap is "
                        "reached — prioritise finishing the task over starting new work.")
            return None

    # -- reporting -------------------------------------------------------------

    def session_report(self, session_id: Optional[str]) -> dict[str, Any]:
        with self._lock:
            self.ledger.sync()  # daily_spent below must reflect other processes
            m = self._sessions.get(session_id or "default")
            if m is None:
                return {}
            return {
                "session_id": m.session_id,
                "duration_s": round(time.time() - m.started_at, 1),
                "spent_usd": round(m.spent_usd, 6),
                "api_calls": m.api_calls,
                "tool_calls": m.tool_calls,
                "blocked_calls": m.blocked_calls,
                "blocked_api_calls": m.blocked_api_calls,
                "session_cap_usd": float(self.config["session_cap_usd"]),
                "daily_spent_usd": round(self.ledger.spent_usd, 6),
                "daily_cap_usd": float(self.config["daily_cap_usd"]),
                "day": self.ledger.day,
            }

    def drop(self, session_id: Optional[str]) -> None:
        with self._lock:
            self._sessions.pop(session_id or "default", None)

    def status_text(self) -> str:
        with self._lock:
            self.ledger.sync()
            cfg = self.config
            lines = [
                f"Budget Leash — today ({self.ledger.day}): ${self.ledger.spent_usd:.2f} of "
                f"${float(cfg['daily_cap_usd']):.2f}; tool calls {self.ledger.tool_calls} of {int(cfg['daily_tool_call_cap'])}.",
            ]
            for m in self._sessions.values():
                lines.append(
                    f"- {m.session_id}: ${m.spent_usd:.2f} of ${float(cfg['session_cap_usd']):.2f}, "
                    f"tool calls {m.tool_calls}, blocked {m.blocked_calls}, llm cut {m.blocked_api_calls}"
                )
            if not self._sessions:
                lines.append("No active sessions tracked.")
            return "\n".join(lines)


def budget_stop_text(reason: str) -> str:
    return (f"Budget limit reached: {reason}. I stopped before doing any more work, "
            "so nothing further was spent. The workspace owner can raise the budget "
            "caps.")


def budget_stop_response(model: Optional[str], reason: str, api_mode: Optional[str] = None) -> Any:
    """A provider-shaped reply that ends the turn with the budget message.

    The ``llm_execution`` middleware returns this *instead of* calling the
    provider once a cap is exhausted. The shape mirrors what the agent
    loop validates and normalizes for the active API mode:

    * ``anthropic_messages`` — a real ``anthropic.types.Message`` when the
      SDK is importable;
    * everything else — a real ``openai.types.chat.ChatCompletion`` when
      the SDK is importable (the OpenAI-compatible modes, the common
      case), otherwise an attribute-identical namespace object.

    If a mode's strict validation still rejects the synthetic reply, the
    loop retries — and every retry is cut off here again, so the provider
    is never reached and the cap holds either way; only the graceful
    wording is mode-dependent. Usage is all-zero: a stopped call spends
    nothing and must meter nothing.
    """
    text = budget_stop_text(reason)
    name = model or "budget-leash"
    if api_mode == "anthropic_messages":
        try:
            from anthropic.types import Message, TextBlock, Usage

            return Message(
                id="budget-leash-stop",
                type="message",
                role="assistant",
                model=name,
                content=[TextBlock(type="text", text=text)],
                stop_reason="end_turn",
                stop_sequence=None,
                usage=Usage(input_tokens=0, output_tokens=0),
            )
        except Exception:
            pass  # fall through to the OpenAI-shaped reply
    try:
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.completion_usage import CompletionUsage

        return ChatCompletion(
            id="budget-leash-stop",
            created=int(time.time()),
            model=name,
            object="chat.completion",
            choices=[Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=text),
            )],
            usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
    except Exception:
        pass
    message = types.SimpleNamespace(role="assistant", content=text, tool_calls=None)
    choice = types.SimpleNamespace(index=0, message=message, finish_reason="stop")
    usage = types.SimpleNamespace(
        prompt_tokens=0, completion_tokens=0, total_tokens=0,
        input_tokens=0, output_tokens=0,
    )
    return types.SimpleNamespace(
        id="budget-leash-stop", created=int(time.time()), model=name,
        object="chat.completion", choices=[choice], usage=usage,
    )


def write_report(report: dict[str, Any]) -> None:
    if not report:
        return
    try:
        path = reports_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(report, default=str) + "\n")
    except Exception:
        pass
