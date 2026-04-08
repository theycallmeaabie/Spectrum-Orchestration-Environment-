# Spectrum Orchestration — OpenEnv RL Benchmark

## Domain: Telecommunications / 5G-6G Network Management

### "The radio spectrum is a warzone. Manage it."

---

## What the AI Does

The AI agent acts as a **Cell Tower Manager**. It is given a geographic area with a
base station (cell tower) and a set of mobile devices (phones, IoT sensors, vehicles).
At every time step the agent must:

1. **Assign a frequency band** (channel) to each connected device.
2. **Set a transmission power level** for each device.

The goal is to **maximise total network throughput** while **minimising
co-channel interference** between devices that share the same band and
maintaining **fairness** so that no single device is starved of bandwidth.

---

## Observation Space

| Component | Shape | Description |
|---|---|---|
| Device positions | `(N, 2)` | x, y coordinates (meters) |
| Device demands | `(N,)` | Requested data rate per device |
| Channel gains | `(N,)` | Path-loss-adjusted gain to tower |
| Current allocations | `(N,)` | Currently assigned channel IDs |
| Current powers | `(N,)` | Currently assigned Tx power |
| Interference map | `(N,)` | Per-device interference level |

Flattened into a single 1-D vector of length `6 * N`.

## Action Space

For each of `N` devices the agent outputs:

- **Channel selection**: integer in `[0, C)` — which frequency band
- **Power level**: continuous in `[P_min, P_max]` dBm

Represented as a `Dict` space or flattened `Box` for compatibility.

---

## Reward & Scoring (0.0 – 1.0)

| Criterion | Weight | Description |
|---|---|---|
| Throughput | 40 % | Sum-rate via Shannon capacity formula |
| Interference penalty | 30 % | Penalty for co-channel devices within range |
| Fairness (Jain's index) | 20 % | Equitable distribution of throughput |
| Power efficiency | 10 % | Prefer lower total transmitted power |

Final score is a weighted sum clipped to `[0.0, 1.0]`.

---

## Difficulty Levels

### Easy — Static Grid
- **Devices**: 10, stationary
- **Channels**: 4
- **Environment**: Open area, no obstacles
- **Mobility**: None

### Medium — Urban Mobility
- **Devices**: 50, moving (random waypoint)
- **Channels**: 8
- **Environment**: Buildings cause shadow fading
- **Mobility**: Devices move 1-5 m/s

### Hard — Dense Metropolitan
- **Devices**: 200, high-speed vehicles + pedestrians
- **Channels**: 16
- **Environment**: Dense urban canyon, Rayleigh fading, handoff between 3 towers
- **Mobility**: 0-30 m/s, non-stationary obstacle map

---

## Why It Matters Now

- **5G/6G rollout** is creating unprecedented spectrum congestion.
- **IoT explosion**: billions of devices competing for limited bandwidth.
- Current heuristic allocators (round-robin, max-SINR) cannot adapt in real-time.
- RL agents can learn **dynamic, adaptive** policies that outperform static algorithms.

---

## Technical Details

- **Path Loss Model**: Log-distance with shadow fading  
  `PL(d) = PL(d₀) + 10·n·log₁₀(d/d₀) + X_σ`
- **SINR Calculation**: Signal-to-Interference-plus-Noise Ratio  
  `SINR_i = P_i · G_i / (Σ_{j≠i, same channel} P_j · G_ji + N₀)`
- **Throughput**: Shannon–Hartley  
  `R_i = B · log₂(1 + SINR_i)`
- **Fairness**: Jain's Fairness Index  
  `J = (Σ R_i)² / (N · Σ R_i²)`
