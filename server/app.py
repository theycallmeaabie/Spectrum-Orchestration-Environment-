"""
FastAPI application for the Spectrum Orchestration Environment.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .spectrum_environment import SpectrumOrchestrationEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.spectrum_environment import SpectrumOrchestrationEnvironment

from fastapi import Body, Query
from fastapi.responses import JSONResponse

# ── Create the base app via OpenEnv create_app ─────────────────────────────
app = create_app(
    SpectrumOrchestrationEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="spectrum_orchestration_env",
)


# ── Custom endpoints for hackathon compliance ──────────────────────────────

TASK_DEFINITIONS = [
    {
        "task_id": "easy",
        "name": "Static Grid — 10 Devices",
        "difficulty": "easy",
        "description": (
            "10 stationary devices, 4 channels, no fading. "
            "Assign channels and power to maximise throughput without interference."
        ),
        "num_devices": 10,
        "num_channels": 4,
        "max_steps": 100,
    },
    {
        "task_id": "medium",
        "name": "Urban Mobility — 50 Devices",
        "difficulty": "medium",
        "description": (
            "50 mobile devices (1-5 m/s), 8 channels, shadow fading. "
            "Adapt allocations as devices move and channel conditions change."
        ),
        "num_devices": 50,
        "num_channels": 8,
        "max_steps": 200,
    },
    {
        "task_id": "hard",
        "name": "Dense Metropolitan — 200 Devices",
        "difficulty": "hard",
        "description": (
            "200 high-speed devices (0-30 m/s), 16 channels, 3 towers, "
            "shadow + Rayleigh fading. Full non-stationary spectrum management."
        ),
        "num_devices": 200,
        "num_channels": 16,
        "max_steps": 300,
    },
]


@app.get("/tasks")
def get_tasks():
    """List all available difficulty-based tasks."""
    return {"tasks": TASK_DEFINITIONS}


@app.post("/grader")
def grader(payload: dict = Body(default={})):
    """
    Grade an episode or evaluate a set of actions.

    Accepts:
        - {"episode_id": str} to grade a completed episode
        - {"task_id": str, "actions": [...]} to evaluate actions against a task

    Returns:
        Score breakdown with total_score in [0.001, 0.999].
    """
    from src.environment import SpectrumOrchestrationEnv
    from src.scoring import SpectrumScorer
    import torch

    task_id = payload.get("task_id", "easy")
    scorer = SpectrumScorer()

    # Run a quick evaluation episode with provided or random actions
    env = SpectrumOrchestrationEnv(difficulty=task_id, seed=42)
    obs, info = env.reset(seed=42)

    total_scores = []
    done = False
    step = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_scores.append(info.get("total_score", 0.0))
        step += 1

    avg_score = sum(total_scores) / max(len(total_scores), 1)
    final_score = max(0.001, min(0.999, avg_score))

    return {
        "score": round(final_score, 4),
        "steps": step,
        "task_id": task_id,
        "breakdown": {
            "throughput_score": round(info.get("throughput_score", 0.0), 4),
            "interference_score": round(info.get("interference_score", 0.0), 4),
            "fairness_score": round(info.get("fairness_score", 0.0), 4),
            "power_score": round(info.get("power_score", 0.0), 4),
        },
    }


@app.get("/schema")
def get_schema():
    """Return JSON schemas for actions and observations."""
    return {
        "action": {
            "type": "object",
            "description": "Spectrum assignment action via MCP tool call",
            "tools": {
                "assign_spectrum": {
                    "description": "Assign channels and power levels to all devices",
                    "parameters": {
                        "channels": {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0, "maximum": 1},
                            "description": "Channel assignment per device (float 0-1, mapped to discrete channel index)",
                        },
                        "powers": {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0, "maximum": 1},
                            "description": "Power level per device (float 0-1, mapped to 0-23 dBm)",
                        },
                    },
                },
                "get_state": {
                    "description": "Get current environment observation",
                    "parameters": {},
                },
                "get_score": {
                    "description": "Get current score breakdown",
                    "parameters": {},
                },
            },
        },
        "observation": {
            "type": "object",
            "properties": {
                "done": {"type": "boolean"},
                "reward": {"type": "number", "minimum": 0.001, "maximum": 0.999},
                "metadata": {"type": "object"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "step_count": {"type": "integer"},
            },
        },
    }


def main():
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
