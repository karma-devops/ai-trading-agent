# AI Trading Agent — Complete Fix Catalogue
# Generated: 2026-07-11
# Status: PRE-EXECUTION (awaiting operator approval)

## Summary
20 issues found across 4 severity levels. 2 CRITICAL fixes will make the agent start.
The rest harden the codebase for production reliability.

---

## CRITICAL FIXES (Agent won't start without these)

### FIX 1: Log writer thread only starts in __main__
**File:** `trading_app.py` lines 3236-3244
**Problem:** Log writer thread is inside `if __name__ == '__main__'` block. Under Gunicorn (wsgi:app), this never runs. `add_console_log()` puts entries in `log_queue` but nothing drains it to `console_logs.json`. Frontend reads empty file → "No activity yet".
**Fix:** Move log writer startup to module level.

```python
# ADD at module level (after line 182, near other global vars):

# Start log writer threads at module level (works under Gunicorn)
log_writer_running = True
log_writer_thread = threading.Thread(target=log_writer_worker, daemon=True)
log_writer_thread.start()
print("✅ Log writer thread started")

backtest_log_writer_running = True
backtest_log_writer_thread = threading.Thread(target=backtest_log_writer_worker, daemon=True)
backtest_log_writer_thread.start()
```

```python
# REMOVE from if __name__ == '__main__' block (lines 3233-3244):
# Start log writer thread
# print("🚀 Starting async log writer...")
# log_writer_running = True
# log_writer_thread = threading.Thread(target=log_writer_worker, daemon=True)
# log_writer_thread.start()
# ... (all of it)
```

**Verification:** After rebuild, console_logs.json should contain entries. Frontend should show logs.

---

### FIX 2: OllamaModel uses native Ollama API, not OpenAI-compatible
**File:** `src/models/ollama_model.py` lines 126-177
**Problem:** Operator's `AI_BASE_URL=https://ollama.com/v1` is an OpenAI-compatible endpoint. But `OllamaModel.initialize_client()` calls `requests.get(f"{self.base_url}/tags")` — the native Ollama API. This URL doesn't exist on ollama.com/v1 → connection fails → `is_available()` returns False → model init fails → agent crashes.
**Fix:** Detect if base_url is OpenAI-compatible (contains `/v1`) and use OpenAI client instead.

```python
# In src/models/ollama_model.py, REPLACE initialize_client():

def initialize_client(self):
    """Initialize the Ollama client connection.
    Detects if base_url is OpenAI-compatible (/v1) and uses OpenAI client."""
    self._is_connected = False
    self._connection_error = None
    self._use_openai_compat = False

    # If base_url contains /v1, use OpenAI-compatible client
    if self.base_url and '/v1' in self.base_url:
        try:
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=self._api_key if hasattr(self, '_api_key') and self._api_key else "ollama",
                base_url=self.base_url
            )
            self._is_connected = True
            self._use_openai_compat = True
            cprint(f"✨ Connected to Ollama via OpenAI-compatible API: {self.base_url}", "green")
            return
        except Exception as e:
            self._connection_error = f"OpenAI client init failed: {e}"
            cprint(f"⚠️ {self._connection_error}", "yellow")
            return

    # Native Ollama API (localhost:11434)
    try:
        response = requests.get(f"{self.base_url}/tags", timeout=5)
        if response.status_code == 200:
            self._is_connected = True
            cprint(f"✨ Connected to Ollama server at {self.base_url}", "green")
        else:
            self._connection_error = f"Ollama API returned status code: {response.status_code}"
    except requests.exceptions.ConnectionError:
        self._connection_error = "Ollama server not running"
        cprint(f"⚠️ Ollama server not running at {self.base_url}", "yellow")
    except Exception as e:
        self._connection_error = str(e)
        cprint(f"⚠️ Ollama connection error: {str(e)}", "yellow")
```

```python
# REPLACE generate_response() to handle both modes:

def generate_response(self, system_prompt, user_content, temperature=0.7, max_tokens=None, **kwargs):
    """Generate a response using Ollama (native or OpenAI-compatible)."""
    if not self._is_connected:
        self.reconnect()
        if not self._is_connected:
            return ModelResponse(
                content="",
                raw_response={"error": f"Ollama not available: {self._connection_error}"},
                model_name=self.model_name,
                usage=None
            )

    # OpenAI-compatible mode
    if self._use_openai_compat:
        try:
            response = self._openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=max_tokens if max_tokens else 4000,
                stream=False
            )
            content = response.choices[0].message.content or ""
            # Strip reasoning tags
            import re
            filtered = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            if '<think>' in filtered:
                filtered = filtered.split('<think>')[0].strip()
            final = filtered if filtered else content
            return ModelResponse(
                content=final,
                raw_response=response,
                model_name=self.model_name,
                usage=getattr(response, 'usage', None)
            )
        except Exception as e:
            cprint(f"❌ Ollama OpenAI-compatible error: {str(e)}", "red")
            return ModelResponse(
                content="",
                raw_response={"error": str(e)},
                model_name=self.model_name,
                usage=None
            )

    # Native Ollama mode (existing code stays the same)
    # ... rest of existing generate_response() ...
```

```python
# ALSO: Update __init__ to accept and store api_key:

def __init__(self, api_key=None, model_name="kimi-k2.7-code", base_url=None):
    self._api_key = api_key  # Store for OpenAI-compatible mode
    self.base_url = base_url or self.DEFAULT_BASE_URL
    self.model_name = model_name
    self._is_connected = False
    self._connection_error = None
    self._available_models = []
    self._use_openai_compat = False
    self._openai_client = None
    super().__init__(api_key="LOCAL_OLLAMA")
    self.initialize_client()
```

**Verification:** After rebuild, console should show "Connected to Ollama via OpenAI-compatible API: https://ollama.com/v1"

---

### FIX 3: Model factory passes wrong env var for ollama base_url
**File:** `src/models/model_factory.py` lines 228-238
**Problem:** When initializing ollama on-demand, factory uses `os.getenv("OLLAMA_BASE_URL")` not `AI_BASE_URL`. Operator's URL is in `AI_BASE_URL`.
**Fix:** Pass the `base_url` parameter from `get_model()` through to the ollama init.

```python
# In get_model(), line 228-238, REPLACE:
if model_type in ("ollama", "ollamafreeapi"):
    model_class = self.MODEL_IMPLEMENTATIONS[model_type]
    default_model = model_name or self.DEFAULT_MODELS.get(model_type)
    init_kwargs = {"model_name": default_model}
    if base_url:
        init_kwargs["base_url"] = base_url
    elif model_type == "ollama":
        # Fallback: try AI_BASE_URL first, then OLLAMA_BASE_URL
        ai_url = os.getenv("AI_BASE_URL", "")
        if ai_url:
            init_kwargs["base_url"] = ai_url
        else:
            init_kwargs["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api")
    if api_key:
        init_kwargs["api_key"] = api_key
    elif model_type == "ollama":
        # Also pass AI_API_KEY for OpenAI-compatible mode
        ai_key = os.getenv("AI_API_KEY", "")
        if ai_key:
            init_kwargs["api_key"] = ai_key
    model_instance = model_class(**init_kwargs)
```

**Verification:** Model factory should pass `https://ollama.com/v1` and the API key to OllamaModel.

---

### FIX 4: config.py hardcodes AI settings, ignores env vars
**File:** `src/config.py` lines 101-103
**Problem:** `AI_MODEL_TYPE = 'ollama'`, `AI_MODEL = "kimi-k2.7-code"`, `AI_BASE_URL = ""` are hardcoded. These are used as fallbacks in trading_agent.py.
**Fix:** Read from env vars.

```python
# REPLACE lines 101-103 in src/config.py:
AI_MODEL_TYPE = os.getenv('AI_PROVIDER', 'ollama')
AI_MODEL = os.getenv('AI_MODEL', 'kimi-k2.7-code')
AI_BASE_URL = os.getenv('AI_BASE_URL', '')
AI_API_KEY = os.getenv('AI_API_KEY', '')
```

**Verification:** `config.py` should pick up operator's env vars.

---

## HIGH FIXES (Functionality broken)

### FIX 5: trading_agent.py hardcodes stale AI model defaults
**File:** `src/agents/trading_agent.py` lines 268-269
**Problem:** `AI_MODEL_TYPE = 'openrouter'`, `AI_MODEL_NAME = 'x-ai/grok-4.1-fast'` — stale, not from env.
**Fix:** Read from env vars or config.

```python
# REPLACE lines 268-269:
from src.config import AI_MODEL_TYPE as CONFIG_AI_MODEL_TYPE, AI_MODEL as CONFIG_AI_MODEL
AI_MODEL_TYPE = os.getenv('AI_PROVIDER', CONFIG_AI_MODEL_TYPE)
AI_MODEL_NAME = os.getenv('AI_MODEL', CONFIG_AI_MODEL)
```

---

### FIX 6: Gunicorn 2 workers causes Flask session issues
**File:** `start.sh` line 6
**Problem:** Flask sessions are in-memory by default. With 2 workers, login on worker 1 may fail on worker 2.
**Fix:** Use 1 worker with 4 threads.

```bash
# REPLACE start.sh line 6:
exec gunicorn -w 1 -b "0.0.0.0:${PORT}" --timeout 120 --worker-class gthread --threads 4 --access-logfile - --error-logfile - wsgi:app
```

---

### FIX 7: load_rbi_jobs() only called in __main__
**File:** `trading_app.py` line 3249
**Problem:** RBI jobs not loaded under Gunicorn.
**Fix:** Since we're disabling backtest, comment this out entirely. See FIX 8.

---

### FIX 8: Disable backtester entirely
**Files:** `trading_app.py` (multiple), `dashboard/templates/index.html`, `dashboard/static/app.js`
**Problem:** Backtest/RBI system is buggy, never worked, not needed in this build.
**Fix:** Comment out all RBI/backtest code and routes.

In `trading_app.py`:
- Comment out `start_rbi_worker()` call
- Comment out all `@app.route('/api/rbi/*')` routes
- Comment out `load_rbi_jobs()` call
- Comment out backtest log writer thread
- Comment out `rbi_worker()` thread start

In `dashboard/templates/index.html`:
- Comment out or remove backtest page link/button if any

In `dashboard/static/app.js`:
- Comment out any backtest-related functions

---

## MEDIUM FIXES (Correctness issues)

### FIX 9: Position sizing too aggressive for small accounts
**File:** `src/config.py` lines 19-20
**Problem:** `TARGET_BALANCE_MULTIPLIER = 50` targets 50x balance. Capped at 92%. With $19.84, position = $18.25 — nearly entire account.
**Fix:** Lower multiplier for small accounts.

```python
# REPLACE lines 19-20:
TARGET_BALANCE_MULTIPLIER = 5   # Target 5x balance (was 50x — too aggressive)
MAX_POSITION_PCT = 0.50         # Max 50% of balance per position (was 92%)
```

---

### FIX 10: Risk limits too tight for small accounts
**File:** `src/config.py` lines 80-81
**Problem:** `MAX_LOSS_USD = 2`, `MAX_GAIN_USD = 3` — would stop after $2 loss on a $19.84 account.
**Fix:** Make proportional or use percentage-based.

```python
# REPLACE lines 72-81:
USE_PERCENTAGE = True  # Use percentage-based limits (better for any account size)

MAX_LOSS_PERCENT = 15   # 15% max loss (was fixed $2)
MAX_GAIN_PERCENT = 50   # 50% max gain (was fixed $3)

# Keep USD limits as fallback (set high so they don't trigger)
MAX_LOSS_USD = 1000
MAX_GAIN_USD = 1000
```

---

### FIX 11: generic_openai_model.py returns None on error
**File:** `src/models/generic_openai_model.py` lines 80-87
**Problem:** Returns `None` on rate limit/credit errors. Caller may not handle None, causing silent analysis failure.
**Fix:** Return ModelResponse with error content.

```python
# REPLACE lines 78-87:
except Exception as e:
    error_str = str(e)
    if "429" in error_str or "rate_limit" in error_str:
        cprint(f"⚠️  Rate limited: {error_str[:80]}", "yellow")
    elif "402" in error_str or "insufficient" in error_str:
        cprint(f"⚠️  Credits insufficient: {error_str[:80]}", "yellow")
    elif "503" in error_str:
        raise e  # Server error, let caller retry
    else:
        cprint(f"❌ Provider error: {error_str[:120]}", "red")
    return ModelResponse(
        content="",
        raw_response={"error": error_str},
        model_name=self.model_name,
        usage=None
    )
```

---

### FIX 12: generic_openai_model.py appends timestamp to user content
**File:** `src/models/generic_openai_model.py` line 54
**Problem:** `f"{user_content}_{timestamp}"` modifies the prompt, could confuse the AI.
**Fix:** Remove timestamp suffix.

```python
# REPLACE line 54:
{"role": "user", "content": user_content}  # Was: f"{user_content}_{timestamp}"
```

---

### FIX 13: ollama_model.py f-string bug
**File:** `src/models/ollama_model.py` line 167
**Problem:** Missing `f` prefix on f-string.
**Fix:**

```python
# REPLACE line 167:
cprint(f"⚠️ Ollama server not running at {self.base_url}", "yellow")  # Was missing f prefix
```

---

## LOW FIXES (Cleanup)

### FIX 14: Remove duplicate DEFAULT_BASE_URL in ollama_model.py
**File:** `src/models/ollama_model.py` lines 22, 106
**Fix:** Remove line 22 (the first definition).

### FIX 15: Remove legacy Solana variables from config.py
**File:** `src/config.py` lines 121-136
**Fix:** Comment out or remove all Solana-specific variables.

### FIX 16: Update stale docstring in trading_agent.py
**File:** `src/agents/trading_agent.py` line 9
**Fix:** Update to reference env vars instead of config.py.

### FIX 17: Remove unused OLLAMA_BASE_URL / GENERIC_OPENAI_BASE_URL from secrets_manager
**File:** `src/utils/secrets_manager.py` (already done in previous commit)

### FIX 18: WebSocket feeds only start in __main__
**File:** `trading_app.py` lines 3128-3231
**Problem:** WebSocket feeds don't start under Gunicorn. App falls back to API polling (acceptable).
**Fix:** Leave as-is. API polling works. WebSocket is optional enhancement.

### FIX 19: Frontend polling interval vs log write interval
**File:** `dashboard/static/app.js` line 38
**Problem:** Frontend polls console every 5s, log writer writes every 2s. Could miss logs if queue overflows.
**Fix:** Acceptable. Queue maxsize=1000, won't overflow in practice.

### FIX 20: ensure src/data directory structure exists on boot
**File:** `trading_app.py` line 154
**Problem:** `DATA_DIR.mkdir()` creates src/data/ but not subdirectories like .cache/logs/.
**Fix:** Already handled in `log_writer_worker()` line 416. No change needed.

---

## EXECUTION ORDER

Phase 1 (CRITICAL — agent starts):
1. FIX 1 — Move log writer to module level
2. FIX 2 — OllamaModel OpenAI-compatible mode
3. FIX 3 — Model factory passes AI_BASE_URL
4. FIX 4 — config.py reads env vars

Phase 2 (HIGH — functionality):
5. FIX 5 — trading_agent.py reads env vars
6. FIX 6 — Gunicorn 1 worker
7. FIX 8 — Disable backtester
8. FIX 7 — (absorbed into FIX 8)

Phase 3 (MEDIUM — correctness):
9. FIX 9 — Position sizing
10. FIX 10 — Risk limits
11. FIX 11 — generic_openai error handling
12. FIX 12 — Remove timestamp from prompt
13. FIX 13 — f-string bug

Phase 4 (LOW — cleanup):
14. FIX 14 — Remove duplicate DEFAULT_BASE_URL
15. FIX 15 — Remove Solana vars
16. FIX 16 — Update docstring

---

## UPSTREAM INSPIRATION (from Moon Dev repos)

### Hyperliquid-Data-Layer-API
- Clean `MoonDevAPI` class with `get_user_positions(address)`, `get_account(address)`
- Could replace direct HL SDK calls for data fetching (more reliable, rate-limited proxy)
- Has liquidation data, whale positions, smart money signals — future enhancement
- API key from moondev.com (future integration)

### Harvard-Algorithmic-Trading-with-AI
- RBI (Research-Backtest-Implement) pattern — we're skipping Backtest for now
- Uses backtesting.py library for strategy validation
- Could integrate later when Eve Engine strategies need validation

### Moon-Dev-Code
- Just links to algotradecamp.com, no useful code