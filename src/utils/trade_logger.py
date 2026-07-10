"""
SQLite trade logger.
Records every trade action to data/trades.db for audit and post-analysis.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent.parent / "data" / "trades.db"


def _ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            side TEXT,
            notional_usd REAL,
            margin_usd REAL,
            leverage INTEGER,
            price REAL,
            pnl_pct REAL,
            reason TEXT,
            strategy TEXT,
            metadata TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_trade(
    symbol: str,
    action: str,
    side: str = "",
    notional_usd: float = 0.0,
    margin_usd: float = 0.0,
    leverage: int = 0,
    price: float = 0.0,
    pnl_pct: float = 0.0,
    reason: str = "",
    strategy: str = "",
    metadata: dict = None,
):
    """Persist one trade record."""
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO trades
        (timestamp, symbol, action, side, notional_usd, margin_usd, leverage, price, pnl_pct, reason, strategy, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            symbol,
            action,
            side,
            notional_usd,
            margin_usd,
            leverage,
            price,
            pnl_pct,
            reason,
            strategy,
            json.dumps(metadata) if metadata else None,
        ),
    )
    conn.commit()
    conn.close()


def get_recent_trades(limit: int = 100):
    """Return recent trade records as a list of dicts."""
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
