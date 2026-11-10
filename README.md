# Budget Leash

**Most agent "budgets" are dashboards — they tell you what you spent after
the money is gone.** The Leash enforces *before* dispatch: when the cap is
reached, the tool call does not run — and the next model call doesn't
happen either.

The scenario it exists for is everyday: unattended agents — cron fleets,
overnight loops, always-on gateways — spend money while nobody is
watching, and a dashboard only tells you afterwards. What the Leash caps
is exactly what it can see: **estimated model spend** (token usage priced
per model) and **tool-call counts**. A tool that bills outside the model
is never priced or gated on a guess — it only counts when you list it
yourself under `tool_costs` with the price you choose. The Leash is that
enforcement as a first-class plugin, on Hermes' own hook surface.

## How it works

Two meters, each tracked per session **and** per UTC day:

| Meter | Source | Notes |
| --- | --- | --- |
| Estimated model spend (USD) | `post_api_request` token usage × per-model price table | Prices configurable per model; a default price covers unlisted models. Cache read/write tokens are priced too (Anthropic `cache_read_input_tokens` / `cache_creation_input_tokens`, OpenAI `prompt_tokens_details.cached_tokens`, Gemini `cached_content_token_count`); set `cache_read` / `cache_write` per model, otherwise each defaults to that model's input price |
| Tool calls | `post_tool_call` (blocked calls are not metered) | Optional flat per-tool USD costs for tools that bill outside the model (paid APIs, device rentals) |

Enforcement points:

- `pre_tool_call` returns a **block** when the session cap, the daily cap,
  either tool-call cap is exhausted — or when a flat-cost tool call would
  itself push the session over its cap.
- The **`llm_execution` middleware** closes the remaining gap: blocking
  tool calls alone still let a capped session burn spend on tool-free
  reasoning turns. Once any cap is exhausted, the middleware answers the
  agent loop with a synthetic provider-shaped "budget exhausted" reply
  and the provider call is **never made** — enforcement covers the model
  itself, not just its tools. On OpenAI-compatible and Anthropic API
  modes the reply is a real SDK response object and the session ends
  with that explanation; on stricter modes a rejected synthetic reply
  just retries into the same cut-off, so the cap holds either way. Cut-off
  calls are counted per session (`blocked_api_calls` in the report).
- `pre_llm_call` injects a **one-time warning at 80%** of a cap, so the
  model can wrap up instead of being cut off mid-task.
- The **daily ledger persists on disk**
  (`~/.hermes/plugin-data/budget-leash/ledger.json`, one per Hermes
  profile), so restarts and parallel processes share one ceiling — a cron
  fleet can't multiply the budget by respawning. Every ledger read and
  write happens under an exclusive file lock (`flock` on
  `ledger.json.lock`) and re-reads the file first, so the CLI
  and the gateway can no longer overwrite each other's totals. A ledger
  left by an older version in the Hermes home root is migrated on first
  load, so a day's totals never reset on upgrade.
- `on_session_end` appends a spend report to
  `~/.hermes/plugin-data/budget-leash/reports.jsonl`. `/leash` shows live status;
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

Cache pricing, per model (defaults to the model's input price when omitted —
a conservative over-estimate, not a silent under-count):

```json
{
  "prices": {"claude-sonnet-4-5": {"input": 3.00, "output": 15.00,
                                   "cache_read": 0.30, "cache_write": 3.75}}
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
costs, the one-time 80% warning, cache-token pricing (configured rates,
input-price default, nested OpenAI details), the ledger race
(interleaved processes sum instead of clobbering, stale readers refresh
before deciding, two real processes racing the ledger), and the
`llm_execution` cut-off (pass-through under the cap, provider never
invoked over it, synthetic reply shape).
