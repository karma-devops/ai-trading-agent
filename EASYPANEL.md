# EasyPanel Deployment Guide

## Quick start (ZIP upload)

1. Download this repository as a ZIP file.
2. In EasyPanel, create a new project.
3. Choose **Deploy from Dockerfile** or **Upload** → upload the ZIP.
4. Expose port `5000`.
5. In the **Environment** tab, add the variables listed in `.env_example`.
6. Start the container.

The health check pings `/api/status` every 30 seconds.

## Required environment variables

| Variable | Why it is needed |
|---|---|
| `FLASK_SECRET_KEY` | Session security. Generate with `python -c "import secrets; print(secrets.token_hex(32))"` |
| `DASHBOARD_USERNAME` | Login username for the web dashboard |
| `DASHBOARD_PASSWORD` | Login password. The app hashes it automatically on first login |
| `HYPER_LIQUID_ETH_PRIVATE_KEY` | HyperLiquid EVM private key used to sign orders |
| `DRY_RUN` | Always set `true` for first tests. `false` goes live |

## Optional but recommended

| Variable | Why it is needed |
|---|---|
| `AGENT_API_KEY` | Long random string. Lets external agents (e.g. Hermes) call `/api/agent/remote/*` |
| `OPENROUTER_API_KEY` | If you select OpenRouter as the AI provider |
| `AI_PROVIDER` / `AI_MODEL` | Default provider/model. Can be changed later in the dashboard |

## Remote agent API

If `AGENT_API_KEY` is set, send it as a header:

```bash
curl -H "X-Agent-API-Key: ***" https://your-domain/api/agent/remote/status
curl -H "X-Agent-API-Key: ***" -X POST https://your-domain/api/agent/remote/start
```

Endpoints:

- `GET /api/agent/remote/status`
- `POST /api/agent/remote/start`
- `POST /api/agent/remote/stop`
- `POST /api/agent/remote/estop`
- `GET/POST /api/agent/remote/settings`
- `GET /api/agent/remote/trades`

## Notes

- Do **not** upload a `.env` file to EasyPanel. Paste each variable into the UI.
- The SQLite trade log and agent state live in `/app/data`, which is mounted from `./data`.
- Logs go to `/app/logs`.
