"""
Spectrum Orchestration Scorer (PyTorch Version)
=============================================
Computes a 0.0–1.0 composite score for the spectrum management task.

Criteria and weights:
    - Throughput        40%   (normalised sum-rate vs theoretical max)
    - Interference      30%   (penalty for co-channel interference)
    - Fairness          20%   (Jain's fairness index)
    - Power efficiency  10%   (prefer lower total Tx power)
"""

from __future__ import annotations
import torch

class SpectrumScorer:
    """Compute a normalised [0, 1] score for spectrum orchestration."""

    def __init__(
        self,
        w_throughput: float = 0.40,
        w_interference: float = 0.30,
        w_fairness: float = 0.20,
        w_power: float = 0.10,
    ):
        self.w_throughput = w_throughput
        self.w_interference = w_interference
        self.w_fairness = w_fairness
        self.w_power = w_power

    def score(
        self,
        throughputs: torch.Tensor,
        interference: torch.Tensor,
        powers: torch.Tensor,
        demands: torch.Tensor,
        p_max: float = 23.0,
        bandwidth: float = 10e6,
    ) -> dict:
        """
        Parameters
        ----------
        throughputs : (N,) tensor  – achieved throughput per device (bps)
        interference : (N,) tensor – interference level per device (linear)
        powers : (N,) tensor – allocated power per device (dBm)
        demands : (N,) tensor – normalised demand per device [0, 1]
        p_max : float – maximum transmit power (dBm)
        bandwidth : float – channel bandwidth (Hz)

        Returns
        -------
        dict with individual and total scores.
        """
        N = len(throughputs)

        # 1. Throughput score
        ref_throughput = 20e6
        total_throughput = torch.sum(throughputs)
        max_total = ref_throughput * N
        throughput_score = torch.clamp(total_throughput / max(max_total, 1e-10), 0.0, 1.0)

        # 2. Interference score
        mean_interf = torch.mean(interference)
        interf_score = torch.exp(-mean_interf * 1e8)
        interf_score = torch.clamp(interf_score, 0.0, 1.0)

        # 3. Fairness score (Jain's index)
        sum_r = torch.sum(throughputs)
        if sum_r < 1e-10:
            fairness_score = torch.tensor(0.0)
        else:
            sum_r2 = torch.sum(throughputs ** 2)
            jain = (sum_r ** 2) / (N * sum_r2) if sum_r2 > 0 else torch.tensor(0.0)
            fairness_score = torch.clamp(jain, 0.0, 1.0)

        # 4. Power efficiency
        mean_power_norm = torch.mean(powers) / max(p_max, 1e-10)
        power_score = torch.clamp(1.0 - mean_power_norm, 0.0, 1.0)

        # Weighted total
        total = (
            self.w_throughput * throughput_score
            + self.w_interference * interf_score
            + self.w_fairness * fairness_score
            + self.w_power * power_score
        )
        # OpenEnv requires rewards strictly in [0.001, 0.999]
        total = torch.clamp(total, 0.001, 0.999)

        return {
            "total_score": total.item(),
            "throughput_score": throughput_score.item(),
            "interference_score": interf_score.item(),
            "fairness_score": fairness_score.item(),
            "power_score": power_score.item(),
        }
