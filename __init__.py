"""Budget Leash — a Hermes plugin.

Hard budget enforcement for agent sessions: estimated model spend and
tool-call counts are metered against a per-session cap and a persistent
per-day cap. ``pre_tool_call`` blocks further tool calls once a cap is
exhausted, and the ``llm_execution`` middleware cuts off the model calls
themselves — without it, a capped session could keep burning spend on
tool-free reasoning turns. See leash.py for the engine and README.md for
rationale and configuration.

Fail-open on internal errors, but fail-closed on budget: when the cap
says no, the call does not run. The hook callbacks swallow exceptions as
usual; the middleware deliberately does not — Hermes' execution chain
already fails open (skips the middleware) when a callback raises before
calling ``next_call``, while a capped session is answered with a
synthetic budget reply and the provider is never invoked.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from . import leash as _engine

logger = logging.getLogger(__name__)

_LEASH = _engine.Leash()


def _session(kwargs: dict[str, Any]) -> Optional[str]:
    return kwargs.get("session_id") or kwargs.get("task_id")


def _on_post_api_request(**kwargs: Any) -> None:
    try:
        _LEASH.on_post_api_request(_session(kwargs), kwargs.get("model"), kwargs.get("usage"))
    except Exception:
        logger.debug("budget-leash: post_api_request failed", exc_info=True)


def _on_pre_tool_call(**kwargs: Any) -> Optional[dict[str, str]]:
    try:
        return _LEASH.on_pre_tool_call(_session(kwargs), str(kwargs.get("tool_name") or ""))
    except Exception:
        logger.debug("budget-leash: pre_tool_call failed", exc_info=True)
        return None


def _on_post_tool_call(**kwargs: Any) -> None:
    try:
        _LEASH.on_post_tool_call(
            _session(kwargs), str(kwargs.get("tool_name") or ""), kwargs.get("status")
        )
    except Exception:
        logger.debug("budget-leash: post_tool_call failed", exc_info=True)


def _on_pre_llm_call(**kwargs: Any) -> Optional[dict[str, str]]:
    try:
        warning = _LEASH.on_pre_llm_call(_session(kwargs))
        return {"context": warning} if warning else None
    except Exception:
        logger.debug("budget-leash: pre_llm_call failed", exc_info=True)
        return None


def _on_llm_execution(**kwargs: Any) -> Any:
    """``llm_execution`` middleware: cut off model calls at the cap.

    While budget remains this is a transparent pass-through. Once a cap
    is exhausted it returns a synthetic provider-shaped reply (see
    ``leash.budget_stop_response``) WITHOUT calling ``next_call`` — the
    provider is never invoked, so no further spend is possible, and the
    agent loop ends the turn with the budget explanation as the reply.
    """
    reason = _LEASH.on_llm_execution(_session(kwargs))
    if reason is None:
        return kwargs["next_call"](kwargs.get("request"))
    logger.info("budget-leash: model call cut off: %s", reason)
    return _engine.budget_stop_response(kwargs.get("model"), reason, kwargs.get("api_mode"))


def _on_session_end(**kwargs: Any) -> None:
    try:
        sid = _session(kwargs)
        _engine.write_report(_LEASH.session_report(sid))
        _LEASH.drop(sid)
    except Exception:
        logger.debug("budget-leash: on_session_end failed", exc_info=True)


def _handle_command(raw_args: str) -> str:
    try:
        argv = (raw_args or "").strip().split()
        if argv and argv[0] == "report":
            import json

            path = _engine.reports_path()
            try:
                lines = path.read_text(encoding="utf-8").strip().splitlines()
                return "Last budget-leash report:\n" + json.dumps(
                    json.loads(lines[-1]), indent=2
                ) if lines else "No budget-leash reports yet."
            except Exception:
                return "No budget-leash reports yet."
        return _LEASH.status_text() + "\n\nUsage: /leash [report]"
    except Exception:
        return "Budget Leash: status unavailable."


def register(ctx) -> None:
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_middleware("llm_execution", _on_llm_execution)
    ctx.register_command(
        "leash",
        handler=_handle_command,
        description="Budget Leash status and last session spend report.",
    )
