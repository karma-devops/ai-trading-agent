# Eve Engine - Full Build Spec & Roadmap

## Source
Operator's complete spec document — Eve Engine withdrawal system, 
6-engine portfolio, monitoring, rotation, alerts, dashboard, UI/UX.

## Status: Phase 0 (current = single engine, basic dashboard working)

## Phase Breakdown

### Phase 1: Core Engine Module (P0)
- [ ] `src/engines/base_engine.py` — Eve Engine core (EMA, ADX, ATR)
- [ ] `src/engines/scalp_engine.py` — 15m aggressive (8/3 trailing)
- [ ] `src/engines/swing_engine.py` — 1h sniper (36/12 trailing)
- [ ] `src/engines/engine_manager.py` — Run 6 engines in parallel
- [ ] `src/config/eve_engines.yaml` — Engine definitions
- [ ] Integration with existing trading_agent.py (parallel mode toggle)

### Phase 2: Multi-Engine Orchestrator (P0)
- [ ] 6 parallel engines with isolated risk
- [ ] Capital allocation per engine (20/20/15/15/20/10)
- [ ] Unified PnL tracking
- [ ] Engine tagging on positions

### Phase 3: Monitoring System (P1)
- [ ] `src/monitoring/performance_tracker.py` — Scoring 0-100
- [ ] `src/monitoring/rotator.py` — Rotation recommendations
- [ ] `src/monitoring/testing_pool.py` — Asset validation pipeline
- [ ] `src/monitoring/alerts.py` — Alert system
- [ ] Status flags (OPTIMAL/HOLD/MONITOR/UNDERPERFORM)

### Phase 4: Withdrawal System (P1)
- [ ] `src/withdrawals/calculator.py` — 50% of profits logic
- [ ] `src/withdrawals/scheduler.py` — 7/14/30 day cycles
- [ ] `src/withdrawals/executor.py` — HL API withdrawal execution
- [ ] Phased withdrawal (0% under $100K, 25% $100-500K, 50% $500K+)
- [ ] Manual buttons (50% profits, withdraw all)

### Phase 5: Dashboard Expansion (P2)
- [ ] Page 1: Overview (total capital, changes, withdrawals, allocation pie)
- [ ] Page 2: Engine Performance (ranked tables, status flags, equity curves)
- [ ] Page 3: Rotation (recommendations, approve/reject, history)
- [ ] Page 4: Asset Testing Pool (candidates, metrics, promote/fail)
- [ ] Page 5: Withdrawals (settings, countdown, manual buttons, history)
- [ ] Page 6: Alerts (active, history, thresholds)
- [ ] Page 7: Settings (allocation editor, withdrawal config, alert config)

### Phase 6: UI/UX Overhaul (P2)
- [ ] Eve Engine color theme (purple/cyan/pink/gold)
- [ ] Visual hierarchy (priority-weighted sizing)
- [ ] Micro-interactions (button feedback, hover glow, data pulse)
- [ ] Console categorization (color-coded by type)
- [ ] Mini charts/sparklines on summary items
- [ ] Modal restructuring (visual grouping)
- [ ] Progress indicators (cycle progress ring, engine status dots)
- [ ] Engine performance table styling
- [ ] Responsive polish
- [ ] Accessibility (focus states, reduced motion, contrast)

## Engine Configuration (Initial 6)

| Engine | Asset | Timeframe | Strategy | Allocation |
|--------|-------|-----------|----------|------------|
| 1 | FARTCOIN | 15m | Scalp Aggressive (8/3) | 20% |
| 2 | HYPE | 15m | Scalp Aggressive (8/3) | 20% |
| 3 | WIF | 15m | Scalp Aggressive (8/3) | 15% |
| 4 | AAVE | 15m | Scalp Aggressive (8/3) | 15% |
| 5 | kPEPE | 1h | Swing Sniper (36/12) | 20% |
| 6 | SOL | 1h | Swing Sniper (36/12) | 10% |

## Key Parameters

### Scalp (15m)
- EMA: 4/9/25
- ADX threshold: 28+
- ATR stop: 1.3x
- Trailing: 8/3 ticks
- Fixed TP: 1.5x ATR
- Time exit: 20 bars max
- Volume confirmation: 1.3x average

### Swing (1h)
- EMA: 6/18/50
- ADX threshold: 18+
- ATR stop: 1.8x
- Trailing: 36/12 ticks
- No fixed TP
- No time exit

## Withdrawal Configuration
- Min capital: $10,000
- Cycle: 7/14/30 days (user toggle)
- Rate: 50% of profits per cycle
- Phased: 0% <$100K, 25% $100-500K, 50% $500K+
- Manual buttons: "50% profits" + "withdraw all"

## Performance Scoring (0-100)
- Return: 30% weight (cap 50% = max)
- Win rate: 20% weight (cap 70% = max)
- Profit factor: 25% weight (cap 3.0 = max)
- Drawdown: 15% weight (20% DD = 0)
- Consistency: 10% weight

## Status Flags
- 75+/70+ = OPTIMAL (keep/increase)
- 60-74/55-69 = HOLD (maintain)
- 45-59/40-54 = MONITOR (consider reduction)
- <45/<40 = UNDERPERFORM (reduce/rotate)

## Alert Thresholds
- Portfolio DD > 15% = CRITICAL
- 5+ consecutive losses = CRITICAL
- Engine DD > 20% = WARNING
- Win rate drop > 10% = WARNING
- PF < 1.5 = WARNING
- No trades 3 days = INFO

## Risk Controls
- Max daily loss: 25%
- Max position size: 30% per engine
- Max consecutive losses: 5 (halt engine)
- Max portfolio DD: 20% (halt all)

## Critical Path to $1M
- Month 1-3: FART 15m compound ($2K → $10K)
- Month 4-6: Add HYPE 15m ($10K → $40K)
- Month 7-9: Add WIF + AAVE ($40K → $100K)
- Month 10-12: Add kPEPE 1h ($100K → $250K)
- Month 13-18: Add SOL 1h ($250K → $600K)
- Month 19-28: Scale withdrawals ($600K → $1M+)