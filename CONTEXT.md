# AI Trading Agent - HyperLiquid (Private)
# Project Context

## Repo
- GitHub: github.com/karma-devops/ai-trading-agent
- Local: /workspace/projects/ai-trading-agent-hl
- Branch: main

## Deployment
- EasyPanel: ai-trader-engine.6cdzen.easypanel.host
- Login: operator / operator
- AGENT_API_KEY: goodgirl999
- Port: 5000
- Base image: python:3.12-slim

## Tech Stack
- Flask + Gunicorn (production server)
- hyperliquid-python-sdk >= 0.24.0 (0.21.0 crashes in Info.__init__ — list index out of range)
- eth-account (key signing)
- pandas-ta 0.4.71b0 (requires Python >= 3.12)
- Frontend: vanilla JS (dashboard/static/app.js), Jinja2 templates

## Architecture
- trading_app.py — main Flask app, all routes, agent loop
- src/exchange_manager.py — HyperLiquid exchange interface
- src/nice_funcs_hyperliquid.py — HL trading functions (buy, sell, positions, balance)
- src/config.py — global config
- src/utils/secrets_manager.py — API keys + trading credentials storage
- src/utils/settings_manager.py — user settings (strategy, tokens, AI model)
- src/utils/tier_manager.py — subscription tiers (currently optional, everyone gets pro)
- dashboard/static/app.js — frontend SPA
- dashboard/templates/index.html — dashboard HTML

## Credential Flow
- Lookup order: secrets JSON (UI-saved) → env vars → legacy fallback
- HYPER_LIQUID_ETH_PRIVATE_KEY: signs orders (MetaMask key or API wallet key)
- ACCOUNT_ADDRESS: MetaMask wallet address for queries (NOT the API wallet address)
- HYPER_LIQUID_KEY: legacy fallback only
- load_secrets_to_env() runs on startup, copies JSON credentials into env vars

## HyperLiquid Key Pitfalls
1. SDK version: must be >= 0.24.0. Older versions crash in Info() constructor.
2. Manual vs Unified account: Manual mode separates spot and perps balances.
   Perps API (userState) returns $0 when USDC is in spot wallet.
   Fix: switch to Unified Account on app.hyperliquid.xyz, or code falls back
   to spotClearinghouseState API.
3. API wallet vs MetaMask: API wallet signs orders but doesn't hold funds.
   Always query with the MetaMask (master) address, never the API wallet address.
4. Address format: HL API accepts both checksummed and lowercase 0x addresses.

## Tier System
- operator = admin (full Pro access)
- When user_tiers.json doesn't exist, everyone gets Pro (multi-user not launched)
- Free (based) tier: default ollama model only, no swarm, no BYOK
- user_tiers.json is gitignored

## Key Decisions
- DRY_RUN: operator sets to false for live trading
- ACTIVE_STRATEGY: engine_v1_3 (Eve Engine v1.3 — PEPE 15m scalp)
- AI_PROVIDER: ollama (glm-5.1 via ollama.com/v1)
- Backtest cruft removed (283MB, 8667 files from src/data/rbi*)