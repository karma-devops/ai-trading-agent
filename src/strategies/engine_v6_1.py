"""
Backward-compatible shim for Engine v6.1.
The full translation now lives in eve_engine_strategies.py.
"""
from .eve_engine_strategies import EngineV6_1Strategy

__all__ = ["EngineV6_1Strategy"]
