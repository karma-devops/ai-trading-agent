# AI Trading Agent - Notes

## Session 2026-07-10: Debug + Harden + Deploy

### Bugs Fixed
1. **Docker build fails at pip install** — pandas-ta 0.4.71b0 requires Python >= 3.12, Dockerfile had 3.10-slim. Fixed: FROM python:3.12-slim.
2. **Strategy dropdown shows [object Object]** — Backend returns dicts {id, name, best_for}, frontend rendered raw object. Fixed: app.js uses strat.id/strat.name.
3. **Settings can't save** — Duplicate active_strategy key in JS object + validator only allowed 2 of 4 strategies. Fixed: removed duplicate, expanded validator.
4. **HL connection fails** — exchange_manager.py read HYPER_LIQUID_KEY (wrong name). Code used HYPER_LIQUID_ETH_PRIVATE_KEY elsewhere. Fixed: unified all lookup paths.
5. **HL SDK crashes** — 0.21.0 crashes in Info.__init__() with "list index out of range". Fixed: upgraded to >=0.24.0.
6. **Balance shows $0** — Operator's account was in Manual mode. USDC in spot wallet, perps API returns $0. Fixed: added spotClearinghouseState fallback. Operator switched to Unified Account.

### Features Added
- HyperLiquid credentials UI in Account > Secrets tab
- /api/trading-credentials GET/POST/DELETE endpoints
- /api/hl-diagnostic endpoint for live debugging
- Helpful info box with step-by-step credential instructions
- Account type warning (Manual vs Unified)
- operator added to ADMIN_USERS
- Free plan locked to default model only
- Tier system optional (gitignored, everyone gets Pro when file absent)
- load_secrets_to_env() now loads trading credentials into env vars

### Cleanup
- Removed 283MB backtest files (src/data/rbi, rbi_v2, rbi_v3, rbi_pp)
- Removed stray HTML backtest reports
- Replaced messy .gitignore with clean version
- src/data/ fully gitignored (runtime data only)

### Commits
- 2fd1538 — Dockerfile python 3.12
- fc8b03a — Strategy dropdown + settings save + HL credentials UI
- da4055b — Credentials loading + operator admin + free plan
- a989c2f — /api/hl-diagnostic endpoint
- 544024f — HL SDK 0.24.0 + 283MB cleanup + tier optional
- 5670b4e — Spot balance fallback + account type info

### Subagent Audit
- Deep codebase audit dispatched (deleg_e37fcfd0)
- Awaiting results for remaining hardening items

### TODO
- [x] Restore original pulse-graph-enhancements flow: engine signals feed LLM and amplify confidence
- [ ] Review subagent audit findings and patch critical issues
- [ ] Update HYPER_LIQUID_ETH_PRIVATE_KEY in EasyPanel env to new API secret
- [ ] Final rebuild and verify live trading works end-to-end

### Current Session Work (2026-07-10 continued)
**Problem:** Engine strategies (`engine_v1_3`, `engine_v1`, `engine_v6_1`) bypassed the LLM entirely. They generated raw technical signals, converted `signal` (0..1) to a percentage, and executed directly. AI was only used as a rubber-stamp for >90% signals.

**Fix in `src/agents/trading_agent.py`:**
1. Added `strategy_signals` parameter to `analyze_market_data()`.
2. Engine path now builds an enriched context in the same shape `StrategyAgent.get_enriched_context()` uses and passes it to `analyze_market_data()`.
3. Added `_amplify_confidence()` helper: blends LLM confidence with engine signal confidence.
   - Agreement boosts confidence toward the engine signal strength.
   - Disagreement dampens confidence.
   - LLM keeps final direction control; only magnitude is adjusted.
   - `NEUTRAL` engine signal provides no amplification.
4. Falls back to raw engine execution when no AI model is available.

**Verification:**
- `python -m py_compile src/agents/trading_agent.py` passes.
- Mock integration test: LLM 60% BUY + engine 85% BUY → final 68% (amplified).
- Mock integration test: LLM 70% BUY + engine 80% SELL → final 50% (dampened, direction still BUY).

**Backup:** `src/agents/trading_agent.py.bak-20260710-194124`