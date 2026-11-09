"""Plain-python tests for the Budget Leash engine.

Run: python3 tests/test_leash.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leash import Leash, Ledger  # noqa: E402


def fresh(tmp: str) -> Leash:
    cfg = {
        "session_cap_usd": 1.00,
        "daily_cap_usd": 2.00,
        "session_tool_call_cap": 5,
        "daily_tool_call_cap": 8,
        "warn_fraction": 0.80,
        "tool_costs": {"paid_api": 0.30},
        "prices": {},
        "default_price": {"input": 1.00, "output": 3.00},
        "disabled": False,
    }
    return Leash(config=cfg, ledger=Ledger(path=Path(tmp) / "ledger.json"))


def test_spend_accumulates_and_session_cap_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        leash = fresh(tmp)
        # 400k input + 100k output at default prices = 0.40 + 0.30 = $0.70
        leash.on_post_api_request("s1", "some-model", {"input_tokens": 400_000, "output_tokens": 100_000})
        assert leash.on_pre_tool_call("s1", "terminal") is None
        # another $0.70 -> $1.40 over the $1.00 session cap
        leash.on_post_api_request("s1", "some-model", {"input_tokens": 400_000, "output_tokens": 100_000})
        decision = leash.on_pre_tool_call("s1", "terminal")
        assert decision is not None and decision["action"] == "block"
        assert "session cap" in decision["message"]
        rep = leash.session_report("s1")
        assert rep["blocked_calls"] == 1 and abs(rep["spent_usd"] - 1.40) < 1e-6
    print("ok: spend accumulates; session cap blocks")


def test_daily_cap_shared_across_sessions_and_persists():
    with tempfile.TemporaryDirectory() as tmp:
        leash = fresh(tmp)
        for sid in ("a", "b"):
            leash.on_post_api_request(sid, "m", {"input_tokens": 700_000, "output_tokens": 100_000})
        # each session spent $1.00; daily total $2.00 hits daily cap for a fresh session
        decision = leash.on_pre_tool_call("fresh-session", "terminal")
        assert decision is not None and "daily cap" in decision["message"]
        # a brand-new Leash (new process) sees the same ledger
        leash2 = fresh(tmp)
        assert leash2.on_pre_tool_call("another", "terminal") is not None
    print("ok: daily cap is shared and persists across processes")


def test_tool_call_caps_and_flat_costs():
    with tempfile.TemporaryDirectory() as tmp:
        leash = fresh(tmp)
        for _ in range(5):
            assert leash.on_pre_tool_call("s", "terminal") is None
            leash.on_post_tool_call("s", "terminal", "ok")
        decision = leash.on_pre_tool_call("s", "terminal")
        assert decision is not None and "tool calls" in decision["message"]
        # blocked calls are not metered
        leash.on_post_tool_call("s", "terminal", "blocked")
        assert leash.session_report("s")["tool_calls"] == 5
        # flat-cost tool: 4 x $0.30 = $1.20 would exceed $1 session cap at the 4th call
        leash2 = fresh(tmp)
        for _ in range(3):
            assert leash2.on_pre_tool_call("t", "paid_api") is None
            leash2.on_post_tool_call("t", "paid_api", "ok")
        assert leash2.on_pre_tool_call("t", "paid_api") is not None
    print("ok: tool-call caps enforced; flat tool costs metered and pre-checked")


def test_warning_at_80_percent_once():
    with tempfile.TemporaryDirectory() as tmp:
        leash = fresh(tmp)
        assert leash.on_pre_llm_call("s") is None
        leash.on_post_api_request("s", "m", {"input_tokens": 500_000, "output_tokens": 100_000})  # $0.80
        warn = leash.on_pre_llm_call("s")
        assert warn is not None and "BUDGET LEASH" in warn
        assert leash.on_pre_llm_call("s") is None  # only once
    print("ok: 80% warning fires once")


def test_cache_tokens_priced_at_configured_rates():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "session_cap_usd": 10.00,
            "daily_cap_usd": 20.00,
            "session_tool_call_cap": 100,
            "daily_tool_call_cap": 1000,
            "warn_fraction": 0.80,
            "tool_costs": {},
            "prices": {"m": {"input": 1.00, "output": 3.00,
                             "cache_read": 0.10, "cache_write": 1.25}},
            "default_price": {"input": 1.00, "output": 3.00},
            "disabled": False,
        }
        leash = Leash(config=cfg, ledger=Ledger(path=Path(tmp) / "ledger.json"))
        # 100k in ($0.10) + 100k out ($0.30) + 1M cache read ($0.10)
        # + 100k cache write ($0.125) = $0.625
        leash.on_post_api_request("s", "m", {
            "input_tokens": 100_000, "output_tokens": 100_000,
            "cache_read_input_tokens": 1_000_000,
            "cache_creation_input_tokens": 100_000,
        })
        rep = leash.session_report("s")
        assert abs(rep["spent_usd"] - 0.625) < 1e-6, rep
        assert abs(rep["daily_spent_usd"] - 0.625) < 1e-6, rep
    print("ok: cache read/write tokens priced at configured rates")


def test_cache_tokens_default_to_input_price():
    with tempfile.TemporaryDirectory() as tmp:
        leash = fresh(tmp)  # default input $1.00/1M, no cache prices set
        # 1M cached reads with no explicit cache price -> $1.00 at input rate,
        # enough to hit the $1.00 session cap (old code priced this $0.00)
        leash.on_post_api_request("s", "m", {"cache_read_input_tokens": 1_000_000})
        assert leash.session_report("s")["spent_usd"] == 1.00
        assert leash.on_pre_tool_call("s", "terminal") is not None
    print("ok: cache tokens default to the input price when unconfigured")


def test_cache_tokens_openai_nested_details():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "session_cap_usd": 10.00, "daily_cap_usd": 20.00,
            "session_tool_call_cap": 100, "daily_tool_call_cap": 1000,
            "warn_fraction": 0.80, "tool_costs": {},
            "prices": {"m": {"input": 1.00, "output": 3.00, "cache_read": 0.10}},
            "default_price": {"input": 1.00, "output": 3.00},
            "disabled": False,
        }
        leash = Leash(config=cfg, ledger=Ledger(path=Path(tmp) / "ledger.json"))
        leash.on_post_api_request("s", "m", {
            "input_tokens": 0, "output_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 500_000},
        })
        assert abs(leash.session_report("s")["spent_usd"] - 0.05) < 1e-6
    print("ok: nested prompt_tokens_details.cached_tokens priced")


def test_ledger_interleaved_processes_do_not_clobber():
    # The exact reviewer scenario: two Leash objects load the ledger at 0,
    # each spends $1.00, and the file must end at $2.00, not $1.00
    # (last-writer-wins). Same-process interleaving exercises apply()'s
    # re-read-under-lock; the multiprocess test below proves it across fds.
    import json

    with tempfile.TemporaryDirectory() as tmp:
        a = fresh(tmp)
        b = fresh(tmp)
        a.on_post_api_request("a", "m", {"input_tokens": 1_000_000})
        b.on_post_api_request("b", "m", {"input_tokens": 1_000_000})
        data = json.loads((Path(tmp) / "ledger.json").read_text())
        assert abs(data["spent_usd"] - 2.00) < 1e-6, data
    print("ok: interleaved ledgers sum instead of clobbering")


def test_stale_reader_sees_other_process_spend():
    with tempfile.TemporaryDirectory() as tmp:
        reader = fresh(tmp)   # loads the ledger at $0, then goes stale
        spender = fresh(tmp)
        spender.on_post_api_request("x", "m", {"input_tokens": 1_000_000})
        spender.on_post_api_request("x", "m", {"input_tokens": 1_000_000})
        # Daily cap is $2.00; without sync() the stale reader would allow.
        decision = reader.on_pre_tool_call("fresh-session", "terminal")
        assert decision is not None and "daily cap" in decision["message"]
    print("ok: pre-call check refreshes the shared ledger before deciding")


def _race_child(tmp: str) -> None:
    leash = fresh(tmp)
    leash.on_post_api_request("worker", "m", {"input_tokens": 1_000_000})


def test_ledger_race_two_real_processes():
    import json
    import multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        ctx = mp.get_context("fork")
        procs = [ctx.Process(target=_race_child, args=(tmp,)) for _ in range(2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0, p.exitcode
        data = json.loads((Path(tmp) / "ledger.json").read_text())
        assert abs(data["spent_usd"] - 2.00) < 1e-6, data
        assert data["api_calls"] == 2, data
    print("ok: two real processes racing the ledger sum to $2.00, not $1.00")


def test_llm_execution_cutoff():
    with tempfile.TemporaryDirectory() as tmp:
        leash = fresh(tmp)
        assert leash.on_llm_execution("s1") is None
        # 2 x $0.70 = $1.40 over the $1.00 session cap
        leash.on_post_api_request("s1", "some-model", {"input_tokens": 400_000, "output_tokens": 100_000})
        leash.on_post_api_request("s1", "some-model", {"input_tokens": 400_000, "output_tokens": 100_000})
        reason = leash.on_llm_execution("s1")
        assert reason and "session cap" in reason, reason
        assert leash.on_llm_execution("s1")  # still capped on the next call
        rep = leash.session_report("s1")
        assert rep["blocked_api_calls"] == 2, rep
        # a different session under the daily cap ($1.40 of $2.00) still passes
        assert leash.on_llm_execution("s2") is None
    print("ok: llm_execution passes under the cap and cuts off over it")


def test_budget_stop_response_shape():
    from leash import budget_stop_response

    resp = budget_stop_response("some-model", "session spend $1.40 reached the session cap $1.00")
    assert resp.choices and resp.choices[0].finish_reason == "stop"
    msg = resp.choices[0].message
    assert msg.role == "assistant" and "BUDGET LEASH" in msg.content
    assert resp.usage.total_tokens == 0
    resp2 = budget_stop_response("m", "some reason", api_mode="anthropic_messages")
    content = getattr(resp2, "content", None)
    if isinstance(content, list):
        assert "BUDGET LEASH" in content[0].text
    else:
        assert "BUDGET LEASH" in resp2.choices[0].message.content
    print("ok: budget stop reply is provider-shaped with zero usage")


class _FakeCtx:
    def __init__(self):
        self.hooks = {}
        self.middleware = {}
        self.commands = {}

    def register_hook(self, name, cb):
        self.hooks[name] = cb

    def register_middleware(self, kind, cb):
        self.middleware[kind] = cb

    def register_command(self, name, handler=None, description=None):
        self.commands[name] = handler


def _load_plugin():
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "budget_leash_pkg", root / "__init__.py")
    spec.submodule_search_paths = [str(root)]
    module = importlib.util.module_from_spec(spec)
    sys.modules["budget_leash_pkg"] = module
    spec.loader.exec_module(module)
    return module


def test_middleware_wiring():
    import os

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["HERMES_HOME"] = tmp
        try:
            plugin = _load_plugin()
            ctx = _FakeCtx()
            plugin.register(ctx)
            assert "llm_execution" in ctx.middleware
            mw = ctx.middleware["llm_execution"]
            calls = []

            def next_call(req=None):
                calls.append(req)
                return "PROVIDER_RESULT"

            out = mw(request={"messages": []}, next_call=next_call,
                     session_id="mw1", model="m", api_mode=None)
            assert out == "PROVIDER_RESULT" and len(calls) == 1
            # default session cap $2.00; 700k in ($0.70) + 500k out ($1.50) = $2.20
            plugin._LEASH.on_post_api_request(
                "mw1", "m", {"input_tokens": 700_000, "output_tokens": 500_000})
            out = mw(request={"messages": []}, next_call=next_call,
                     session_id="mw1", model="m", api_mode=None)
            assert len(calls) == 1, "provider must not be invoked once the cap is hit"
            assert "BUDGET LEASH" in out.choices[0].message.content
            assert plugin._LEASH.session_report("mw1")["blocked_api_calls"] == 1
        finally:
            os.environ.pop("HERMES_HOME", None)
    print("ok: middleware passes through under the cap, cuts off over it")


if __name__ == "__main__":
    test_spend_accumulates_and_session_cap_blocks()
    test_daily_cap_shared_across_sessions_and_persists()
    test_tool_call_caps_and_flat_costs()
    test_warning_at_80_percent_once()
    test_cache_tokens_priced_at_configured_rates()
    test_cache_tokens_default_to_input_price()
    test_cache_tokens_openai_nested_details()
    test_ledger_interleaved_processes_do_not_clobber()
    test_stale_reader_sees_other_process_spend()
    test_ledger_race_two_real_processes()
    test_llm_execution_cutoff()
    test_budget_stop_response_shape()
    test_middleware_wiring()
    print("ALL TESTS PASSED")
