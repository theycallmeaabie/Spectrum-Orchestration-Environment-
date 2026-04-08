"""
OpenEnv Sample Inference Script — Spectrum Orchestration
=========================================================
Follows the Pre-Submission Checklist strictly:
  - Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN, LOCAL_IMAGE_NAME
  - Defaults set only for API_BASE_URL and MODEL_NAME (not HF_TOKEN)
  - All LLM calls use the OpenAI client configured via these variables
  - Stdout logs follow the required format: START / STEP / END
"""

import os
import sys
import json
import random

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from client import SpectrumEnv

# ── 1. Environment variables ───────────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL",  "https://api.openai.com/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",    "gpt-4o")
HF_TOKEN         = os.getenv("HF_TOKEN")                  # no default
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")           # optional

# ── 2. OpenAI client (all LLM calls go through this) ──────────────────────────
llm_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "dummy-token",
)

# ── 3. Environment connection ──────────────────────────────────────────────────
def get_env_base_url() -> str:
    """Resolve the environment server URL."""
    if LOCAL_IMAGE_NAME:
        return "http://localhost:8000"
    hf_space = os.getenv("SPACE_ID", "")
    if hf_space:
        return f"https://{hf_space.replace('/', '-')}.hf.space"
    return "http://localhost:8000"


# ── 4. Task configs ───────────────────────────────────────────────────────────
TASK_CONFIGS = {
    "easy":   {"num_devices": 10,  "max_steps": 100},
    "medium": {"num_devices": 50,  "max_steps": 200},
    "hard":   {"num_devices": 200, "max_steps": 300},
}


def get_llm_action(num_devices: int, step_num: int, last_result: dict | None) -> tuple[list, list]:
    """Ask the LLM for channel and power assignments."""
    context = ""
    if last_result:
        context = (
            f"\nPrevious step result: "
            f"reward={last_result.get('reward', '?')}, "
            f"throughput={last_result.get('throughput_score', '?')}, "
            f"interference={last_result.get('interference_score', '?')}, "
            f"fairness={last_result.get('fairness_score', '?')}. "
            f"Improve your allocation."
        )

    prompt = (
        f"You are a 5G spectrum management agent. "
        f"There are {num_devices} wireless devices that need channel and power assignments. "
        f"This is step {step_num}.{context}\n\n"
        f"Each device needs:\n"
        f"- channel: float in [0, 1] (mapped to discrete channel index)\n"
        f"- power: float in [0, 1] (mapped to 0-23 dBm)\n\n"
        f"Strategy tips:\n"
        f"- Spread devices across different channels to reduce interference\n"
        f"- Use lower power when possible for efficiency\n"
        f"- Ensure fair throughput distribution\n\n"
        f"Respond with ONLY valid JSON, no markdown:\n"
        f'{{"channels": [<{num_devices} floats>], "powers": [<{num_devices} floats>]}}'
    )

    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an RL agent for 5G spectrum management. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
        )
        content = response.choices[0].message.content
        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        parsed = json.loads(content)
        channels = [float(x) for x in parsed["channels"][:num_devices]]
        powers = [float(x) for x in parsed["powers"][:num_devices]]
        return channels, powers
    except Exception:
        # Fallback: spread channels evenly, moderate power
        channels = [(i / num_devices) for i in range(num_devices)]
        powers = [0.4] * num_devices
        return channels, powers


def run_episode(env, task_id: str, max_steps: int, num_devices: int) -> dict:
    """Run a single episode on a given difficulty."""
    print(f"STEP: Resetting environment for task={task_id}", flush=True)
    env.reset()

    last_result = None
    total_reward = 0.0
    steps_taken = 0

    for step in range(max_steps):
        # Get action from LLM (with fallback to heuristic)
        channels, powers = get_llm_action(num_devices, step, last_result)

        # Pad/trim to correct device count
        channels = (channels + [0.5] * num_devices)[:num_devices]
        powers = (powers + [0.4] * num_devices)[:num_devices]

        try:
            result = env.call_tool("assign_spectrum", channels=channels, powers=powers)
            last_result = result
            reward = result.get("reward", 0.0)
            total_reward += reward
            steps_taken += 1

            if step % 10 == 0 or result.get("done", False):
                print(
                    f"STEP: [{task_id}] step={step} "
                    f"reward={reward:.4f} "
                    f"score={result.get('total_score', '?')} "
                    f"throughput={result.get('throughput_score', '?')} "
                    f"interference={result.get('interference_score', '?')}",
                    flush=True,
                )

            if result.get("done", False):
                break
        except Exception as e:
            print(f"STEP: [{task_id}] step={step} error — {e}", flush=True)
            break

    avg_reward = total_reward / max(steps_taken, 1)
    print(
        f"STEP: [{task_id}] Episode complete — "
        f"steps={steps_taken} avg_reward={avg_reward:.4f}",
        flush=True,
    )
    return {
        "task_id": task_id,
        "steps": steps_taken,
        "total_reward": round(total_reward, 4),
        "avg_reward": round(avg_reward, 4),
        "final_score": last_result.get("total_score", 0.0) if last_result else 0.0,
    }


def main():
    print("START", flush=True)

    base_url = get_env_base_url()
    print(f"STEP: Connecting to {base_url}", flush=True)

    results = []

    try:
        with SpectrumEnv(base_url=base_url) as env:
            print("STEP: Connected. Listing tools...", flush=True)
            try:
                tools = env.list_tools()
                print(f"STEP: Tools — {[t.name for t in tools]}", flush=True)
            except Exception:
                print("STEP: Tool listing skipped, proceeding with known tools", flush=True)

            # Run all 3 difficulty levels
            for task_id, cfg in TASK_CONFIGS.items():
                print(f"STEP: === Starting {task_id.upper()} difficulty ===", flush=True)
                result = run_episode(
                    env,
                    task_id=task_id,
                    max_steps=cfg["max_steps"],
                    num_devices=cfg["num_devices"],
                )
                results.append(result)

    except Exception as e:
        print(f"STEP: Connection error — {e} (is the server running?)", flush=True)

    # Summary
    print("STEP: === RESULTS SUMMARY ===", flush=True)
    for r in results:
        print(
            f"STEP: {r['task_id']:>6s} | "
            f"steps={r['steps']:>3d} | "
            f"avg_reward={r['avg_reward']:.4f} | "
            f"final_score={r['final_score']:.4f}",
            flush=True,
        )

    print("END", flush=True)


if __name__ == "__main__":
    main()
