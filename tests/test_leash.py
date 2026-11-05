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


if __name__ == "__main__":
    test_spend_accumulates_and_session_cap_blocks()
    test_daily_cap_shared_across_sessions_and_persists()
    test_tool_call_caps_and_flat_costs()
    test_warning_at_80_percent_once()
    print("ALL TESTS PASSED")
