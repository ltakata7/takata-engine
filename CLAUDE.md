# takata-engine

> **Active repo: `takata-engine`** — state this at the top of any session summary. Changes here propagate to three consumers, so make it obvious.

## Scope (what lives here)

Shared analytics library. Consumed via `git+https` by `takata-trading`, `takata-wealth`, and `sympatheia-os`.

Owns: macro cycle classifier, BCB/FRED feeds, position sizing, and Claude-powered agents (`risk_narrator`, `macro_flash`, `premarket_briefing`).

## Out of scope — do NOT add here

- Anything app-specific: UI, FastAPI routers, database schemas, trading loops, broker bridges, client/account management. Those live in the consumer repos.
- Secrets, API keys, broker credentials. This library is imported into multiple apps — keep it pure.
- Heavy runtime dependencies that only one consumer needs. If only trading needs it, it belongs in trading.

## Cross-repo guardrails

- **Every change here is a cross-repo change.** A push to `main` does not auto-propagate; consumers pin by commit SHA (or should). Still: assume a breaking API change affects three apps and write the deprecation path accordingly.
- Before renaming or removing a public symbol, grep the three consumer repos (`~/takata-trading`, `~/takata-wealth`, `~/SYMPATHEIA_OS`) for callers.
- Tests should cover the public API surface, not internal helpers. Consumers depend on the public surface; that's the contract.

## Operational notes

- Tests: `pytest tests/ -q`.
- Consumers pick up changes via `pip install -e .` (editable local) or by bumping the git SHA in their `pyproject.toml`.
