# Budget Leash

**Most agent "budgets" are dashboards — they tell you what you spent after
the money is gone.** The Leash enforces *before* dispatch: when the cap is
reached, the tool call does not run.

The scenario it exists for is not hypothetical: an unattended agent once
hit **7 unapproved Stripe endpoints and double-charged a customer within
11 minutes**. The fix at the time was an external wrapper. The Leash is
that fix as a first-class plugin, on Hermes' own hook surface.

## How it works

Two meters, each tracked per session **and** per UTC day:

| Meter | Source | Notes |
| --- | --- | --- |
| Estimated model spend (USD) | `post_api_request` token usage × per-model price table | Prices configurable per model; a default price covers unlisted models |
| Tool calls | `post_tool_call` (blocked calls are not metered) | Optional flat per-tool USD costs for tools that bill outside the model (paid APIs, device rentals) |

Enforcement points:

- `pre_tool_call` returns a **block** when the session cap, the daily cap,
  either tool-call cap is exhausted — or when a flat-cost tool call would
  itself push the session over its cap.
- `pre_llm_call` injects a **one-time warning at 80%** of a cap, so the
  model can wrap up instead of being cut off mid-task.
- The **daily ledger persists on disk** (`~/.hermes/budget-leash-ledger.json`),
  so restarts and parallel processes share one ceiling — a cron fleet
  can't multiply the budget by respawning.
- `on_session_end` appends a spend report to
  `~/.hermes/budget-leash-reports.jsonl`. `/leash` shows live status;
  `/leash report` prints the last session's numbers.

A blocked agent can still write its final answer — plain replies are not
tool calls — and the block message tells it to explain the budget stop to
the user.

## Configuration

Defaults: session cap $2.00, daily cap $10.00, 200 tool calls/session,
1000/day, warn at 80%. Override in `~/.hermes/budget-leash.json`:

```json
{
  "session_cap_usd": 5.00,
  "daily_cap_usd": 25.00,
  "prices": {"gemini-3.5-flash-lite": {"input": 0.10, "output": 0.40}},
  "tool_costs": {"rent_device": 1.50}
}
```

Environment overrides: `LEASH_SESSION_CAP_USD`, `LEASH_DAILY_CAP_USD`,
`LEASH_SESSION_TOOL_CALLS`, `LEASH_DAILY_TOOL_CALLS`. `LEASH_DISABLE=1`
is a kill switch.

## Safety & scope

- **Fail-closed on budget, fail-open on bugs:** hook exceptions are
  swallowed (a Leash bug never breaks the agent loop), but a reached cap
  always blocks.
- Spend is an *estimate* from reported token usage and your price table —
  exact billing remains the provider's invoice. For free models, set
  prices to 0 and enforce purely on call counts.
- No credentials are read; no network calls are made.

## Install

Drop this directory into `~/.hermes/plugins/budget-leash/`, enable it
(`hermes plugins enable budget-leash`), and restart Hermes.

## Tests

`python3 tests/test_leash.py` — spend accumulation, session/daily cap
blocking, ledger persistence across processes, tool-call caps, flat tool
costs, and the one-time 80% warning.
