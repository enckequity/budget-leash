"""Budget Leash engine — hard spend caps for Hermes sessions.

Most agent "budgets" are dashboards: they tell you what you spent after the
money is gone. The Leash enforces *before* dispatch. It accumulates two
meters per session and per UTC day:

  * estimated model spend — token usage from ``post_api_request`` priced
    with a configurable per-model table (USD per 1M tokens), and
  * tool calls — counted at ``post_tool_call``, with optional flat per-tool
    costs for tools that bill outside the model (paid APIs, rentals).

``pre_tool_call`` compares the meters against a per-session cap and a
persistent per-day cap and returns a block decision when either is
exhausted. The day ledger lives on disk and survives restarts, so an
unattended cron fleet shares one daily ceiling instead of one per process.
At 80% of a cap the model gets a one-time context warning; at 100% tool
calls stop — the agent can still write its final answer explaining why,
because plain replies are not tool calls.

Fail-open on internal errors (the plugin layer swallows exceptions), but
fail-closed on budget: when the cap says no, the call does not run.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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

_ENV_MAP = {
    "LEASH_SESSION_CAP_USD": ("session_cap_usd", float),
    "LEASH_DAILY_CAP_USD": ("daily_cap_usd", float),
    "LEASH_SESSION_TOOL_CALLS": ("session_tool_call_cap", int),
    "LEASH_DAILY_TOOL_CALLS": ("daily_tool_call_cap", int),
}


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


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
    """Persistent per-day totals, shared across processes via the file."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (hermes_home() / "budget-leash-ledger.json")
        self.day = _today()
        self.spent_usd = 0.0
        self.tool_calls = 0
        self.api_calls = 0
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
            if data.get("day") == self.day:
                self.spent_usd = float(data.get("spent_usd", 0.0))
                self.tool_calls = int(data.get("tool_calls", 0))
                self.api_calls = int(data.get("api_calls", 0))
        except Exception:
            pass

    def roll(self) -> None:
        today = _today()
        if today != self.day:
            self.day = today
            self.spent_usd = 0.0
            self.tool_calls = 0
            self.api_calls = 0
            self.save()

    def save(self) -> None:
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "day": self.day,
                "spent_usd": round(self.spent_usd, 6),
                "tool_calls": self.tool_calls,
                "api_calls": self.api_calls,
            }))
            tmp.replace(self.path)
        except Exception:
            pass


@dataclass
class SessionMeter:
    session_id: str
    spent_usd: float = 0.0
    tool_calls: int = 0
    api_calls: int = 0
    blocked_calls: int = 0
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

    def _price_for(self, model: Optional[str]) -> tuple[float, float]:
        prices = self.config.get("prices", {})
        entry = prices.get(model or "") or self.config["default_price"]
        return float(entry.get("input", 0.0)), float(entry.get("output", 0.0))

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
            self.ledger.roll()
            m = self._meter(session_id)
            m.api_calls += 1
            self.ledger.api_calls += 1
            if isinstance(usage, dict):
                in_tok = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
                out_tok = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
                pin, pout = self._price_for(model)
                cost = in_tok / 1e6 * pin + out_tok / 1e6 * pout
                m.spent_usd += cost
                self.ledger.spent_usd += cost
            self.ledger.save()

    def on_pre_tool_call(self, session_id: Optional[str], tool_name: str) -> Optional[dict[str, str]]:
        """Return a block decision when a cap is exhausted, else None."""
        if self.config.get("disabled"):
            return None
        with self._lock:
            self.ledger.roll()
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
                                "The user can raise the caps in ~/.hermes/budget-leash.json."),
                }
            return None

    def on_post_tool_call(self, session_id: Optional[str], tool_name: str, status: Any) -> None:
        if self.config.get("disabled"):
            return
        with self._lock:
            if status == "blocked":
                return  # blocked calls never ran; don't meter them
            self.ledger.roll()
            m = self._meter(session_id)
            m.tool_calls += 1
            self.ledger.tool_calls += 1
            flat = float(self.config.get("tool_costs", {}).get(tool_name, 0.0))
            if flat > 0:
                m.spent_usd += flat
                self.ledger.spent_usd += flat
            self.ledger.save()

    def on_pre_llm_call(self, session_id: Optional[str]) -> Optional[str]:
        """One-time 80% warning so the model can wrap up before the hard stop."""
        if self.config.get("disabled"):
            return None
        with self._lock:
            m = self._meter(session_id)
            if m.warned:
                return None
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
            cfg = self.config
            lines = [
                f"Budget Leash — today ({self.ledger.day}): ${self.ledger.spent_usd:.2f} of "
                f"${float(cfg['daily_cap_usd']):.2f}; tool calls {self.ledger.tool_calls} of {int(cfg['daily_tool_call_cap'])}.",
            ]
            for m in self._sessions.values():
                lines.append(
                    f"- {m.session_id}: ${m.spent_usd:.2f} of ${float(cfg['session_cap_usd']):.2f}, "
                    f"tool calls {m.tool_calls}, blocked {m.blocked_calls}"
                )
            if not self._sessions:
                lines.append("No active sessions tracked.")
            return "\n".join(lines)


def write_report(report: dict[str, Any]) -> None:
    if not report:
        return
    try:
        path = hermes_home() / "budget-leash-reports.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(report, default=str) + "\n")
    except Exception:
        pass
