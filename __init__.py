"""
Spectrum Orchestration Environment
===================================
A Gymnasium-compatible RL environment for 5G/6G dynamic spectrum management.
Built on Meta's OpenEnv framework for the Meta PyTorch Hackathon.

Quick start:
    from spectrum_orchestration_env import SpectrumEnv

    with SpectrumEnv(base_url="http://localhost:8000").sync() as env:
        env.reset()
        result = env.call_tool("assign_spectrum", channels=[0.5]*10, powers=[0.5]*10)
        print(result)

Or use the Gymnasium environment directly for training:
    from src.environment import SpectrumOrchestrationEnv

    env = SpectrumOrchestrationEnv(difficulty="easy", seed=42)
    obs, info = env.reset()
"""

import sys
import os as _os
_root = _os.path.dirname(_os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.environment import SpectrumOrchestrationEnv, Difficulty  # noqa: E402

def __getattr__(name):
    """Lazy-load SpectrumEnv so openenv-core is not required at import time."""
    if name == "SpectrumEnv":
        from client import SpectrumEnv
        return SpectrumEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["SpectrumEnv", "SpectrumOrchestrationEnv", "Difficulty"]
__version__ = "0.1.0"
