---
title: Spectrum Orchestration Env
emoji: 📡
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
---

# Spectrum Orchestration — OpenEnv RL Benchmark


> **Domain**: Telecommunications / 5G-6G Dynamic Spectrum Management  
> **Track**: [Meta PyTorch Hackathon — OpenEnv](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon#open-ev)

An RL environment where an AI agent manages radio frequency spectrum allocation
for mobile devices, balancing throughput, interference, fairness, and power
efficiency — scored on a 0.001 – 0.999 scale with partial credit.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run tests

```bash
python -m pytest tests/ -v
```

### 3. Start the OpenEnv server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 4. Run the inference script (connects to running server)

```bash
python inference.py
```

### 5. Train a baseline PPO agent (local, no server needed)

```bash
python src/train.py --difficulty easy --episodes 300 --random-baseline
```

### 6. Use the environment in your own code

```python
from src.environment import SpectrumOrchestrationEnv

env = SpectrumOrchestrationEnv(difficulty="medium", seed=42)
obs, info = env.reset()

for step in range(200):
    action = env.action_space.sample()  # replace with your agent
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}  Score: {info['total_score']:.3f}")
    if terminated or truncated:
        break
```

### 7. Connect via OpenEnv MCP client

```python
from client import SpectrumEnv

with SpectrumEnv(base_url="http://localhost:8000") as env:
    env.reset()
    result = env.call_tool("assign_spectrum", channels=[0.5]*10, powers=[0.5]*10)
    print(result)
```

---

## Environment Design

| Component | Details |
|---|---|
| **Observation** | Device positions, demands, channel gains, current allocations, interference (flat vector, `6xN`) |
| **Action** | Channel assignment + power level per device (flat vector, `2xN`) |
| **Reward** | Composite score: 40% throughput + 30% interference + 20% fairness + 10% power efficiency |
| **Range** | Strictly [0.001, 0.999] — deterministic, Shannon-formula grounded |
| **Termination** | Fixed horizon (100 / 200 / 300 steps by difficulty) |

### Difficulty Levels

| Level | Devices | Channels | Towers | Mobility | Fading | Steps |
|---|---|---|---|---|---|---|
| Easy | 10 | 4 | 1 | None | None | 100 |
| Medium | 50 | 8 | 1 | 1-5 m/s | Shadow | 200 |
| Hard | 200 | 16 | 3 | 0-30 m/s | Shadow + Rayleigh | 300 |

---

## Scoring Breakdown

| Criterion | Weight | Formula |
|---|---|---|
| **Throughput** | 40% | `sum(B * log2(1+SINR)) / reference` |
| **Interference** | 30% | `exp(-mean_interference)` |
| **Fairness** | 20% | Jain's Fairness Index |
| **Power Efficiency** | 10% | `1 - mean(P)/P_max` |

---

## Physics Model

- **Path Loss**: Log-distance with configurable shadow fading  
  `PL(d) = PL(d0) + 10 * n * log10(d/d0) + X_sigma`
- **SINR**: Signal-to-Interference-plus-Noise Ratio per device  
  `SINR_i = P_i * G_i / (sum(P_j * G_ji, j!=i same channel) + N0)`
- **Throughput**: Shannon-Hartley capacity theorem  
  `R_i = B * log2(1 + SINR_i)`
- **Fairness**: Jain's Fairness Index  
  `J = (sum(R_i))^2 / (N * sum(R_i^2))`

---

## Project Structure

```
spectrum-orchestration-env/
├── client.py              # OpenEnv MCP client (MCPToolClient)
├── inference.py           # Required inference script (LLM agent + heuristic fallback)
├── openenv.yaml           # OpenEnv environment manifest
├── pyproject.toml         # Dependencies and build config
├── requirements.txt       # pip requirements
├── Dockerfile             # Multi-stage OpenEnv container
├── design.md              # Detailed environment design document
├── __init__.py            # Package entry point
├── server/
│   ├── __init__.py
│   ├── app.py             # FastAPI server (create_app + custom /tasks, /grader, /schema)
│   └── spectrum_environment.py  # MCPEnvironment wrapper with MCP tools
├── src/
│   ├── __init__.py
│   ├── environment.py     # Gymnasium RL environment (PyTorch physics)
│   ├── scoring.py         # Composite scorer (0.001-0.999)
│   └── train.py           # PPO baseline training script
└── tests/
    └── test_environment.py  # Pytest suite
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Environment metadata |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List difficulty levels |
| `POST` | `/reset` | Reset environment |
| `POST` | `/step` | Execute action via MCP |
| `GET` | `/state` | Current episode state |
| `POST` | `/grader` | Evaluate episode score |
| `GET` | `/schema` | JSON schemas for action/observation |

### MCP Tools

| Tool | Description |
|---|---|
| `assign_spectrum(channels, powers)` | Assign frequency bands and power levels to all devices |
| `get_state()` | Return current observation summary |
| `get_score()` | Return current score breakdown |

---

## Deploy to Hugging Face Spaces

```bash
# Build Docker image
docker build -t spectrum-env .

# Test locally
docker run -p 8000:8000 spectrum-env

# Push to HF Spaces (via HF web UI or CLI)
```

---

## Why This Matters

- **5G/6G rollout** is creating unprecedented spectrum congestion
- **IoT explosion**: billions of devices competing for limited bandwidth
- Current heuristic allocators cannot adapt in real-time
- RL agents can learn dynamic, adaptive policies that outperform static algorithms
- This environment is **non-stationary** (devices move, channels fade) — a genuine RL challenge

---

## Tech Stack

- **Python 3.10+**
- **PyTorch** — neural network training and all radio physics simulation
- **Gymnasium** — RL environment interface
- **OpenEnv** — Meta's framework for agentic environment deployment
- **FastAPI** — HTTP/WebSocket server
- **FastMCP** — MCP tool registration

---

## License

MIT
