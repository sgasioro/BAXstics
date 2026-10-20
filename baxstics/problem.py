"""The hydrogenolysis problem on abbot.

Designs are reaction conditions (``Time``, ``Temperature``, ``Pressure``) from a
fixed candidate grid; the measured properties are ``Yield`` (%) and ``Median``
(median carbon number of the liquid product). The goal is, for each band of
``Median``, the conditions with the highest ``Yield`` (:class:`TopKPerBin`).

abbot fits one GP per output (BoTorch's default: an ARD RBF kernel whose
dimension-scaled lengthscale prior keeps sparse, partly flat data fittable; inputs
scaled to the unit cube by the spec bounds) and runs ``BAXPolicy`` over the grid. Goals see
outputs in physical units, so no output scalers are needed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from abbot import (
    BAXPolicy,
    Campaign,
    ContinuousDomain,
    ModelListConstructor,
    OutputSpec,
    ProblemSpec,
)

XCOLS = ["Time", "Temperature", "Pressure"]
YCOLS = ["Yield", "Median"]
# Physical ranges of the reactor conditions (hours, deg C, bar).
X_BOUNDS = {"Time": (1.0, 24.0), "Temperature": (200.0, 250.0), "Pressure": (18.0, 50.0)}


def make_spec(bounds: Mapping[str, tuple[float, float]] = X_BOUNDS) -> ProblemSpec:
    """Three continuous conditions and two observed (not optimized) outputs: the
    goal, not an objective direction, says what to look for."""
    return ProblemSpec(
        domain=[
            ContinuousDomain(name=name, lower=float(bounds[name][0]), upper=float(bounds[name][1]))
            for name in XCOLS
        ],
        outputs=[OutputSpec(name=name, direction="OBSERVE") for name in YCOLS],
    )


class TopKPerBin:
    """For each ``[lo, hi]`` band of ``Median``, the ``k`` candidates with the highest
    ``Yield``. A band is widened by ``epsilon`` on both sides; empty bands are
    skipped. Port of the multibax ``TopKBinAlgorithm``.
    """

    def __init__(
        self,
        bins: Sequence[tuple[float, float]],
        k: int = 1,
        epsilon: float = 0.0,
        value: int = 0,
        by: int = 1,
    ) -> None:
        self.bins = [(float(lo), float(hi)) for lo, hi in bins]
        self.k = k
        self.epsilon = epsilon
        self.value = value  # output column ranked (Yield)
        self.by = by  # output column binned (Median)

    def identify_subspace(self, f: torch.Tensor, X: torch.Tensor | None = None) -> list[int]:
        f = f.reshape(f.shape[0], -1)
        ids: set[int] = set()
        for lo, hi in self.bins:
            in_bin = (f[:, self.by] >= lo - self.epsilon) & (f[:, self.by] <= hi + self.epsilon)
            idx = in_bin.nonzero(as_tuple=True)[0]
            if len(idx):
                order = torch.argsort(f[idx, self.value], descending=True)
                ids.update(idx[order[: self.k]].tolist())
        return sorted(ids)


def make_campaign(
    grid: torch.Tensor,
    goal,
    *,
    strategy: str = "SwitchBAX",
    n_posterior_samples: int = 10,
    bounds: Mapping[str, tuple[float, float]] = X_BOUNDS,
) -> Campaign:
    """A campaign proposing from ``grid`` ``(N, 3)``, columns in ``XCOLS`` order."""
    spec = make_spec(bounds)
    policy = BAXPolicy(
        goal,
        strategy=strategy,
        candidates=grid.double(),
        n_posterior_samples=n_posterior_samples,
    )
    model = ModelListConstructor(bounds=spec.bounds_tensor())
    return Campaign(spec, policy=policy, model_constructor=model)


def grid_index(grid: torch.Tensor, x: Mapping[str, float] | torch.Tensor) -> int:
    """Row of ``grid`` holding design ``x`` (a suggested candidate or a tensor)."""
    if isinstance(x, Mapping):
        x = torch.tensor([float(x[name]) for name in XCOLS], dtype=torch.double)
    hits = torch.isclose(grid.double(), x.double().reshape(1, -1)).all(dim=-1).nonzero()
    if len(hits) == 0:
        raise ValueError(f"design {x.tolist()} is not on the candidate grid")
    return int(hits[0])


def raw_mean_std(campaign: Campaign, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior mean and std ``(N, 2)`` at ``X`` in physical units, from the
    campaign's current GP (both outputs are OBSERVE, so no sign flip applies)."""
    model = campaign.current_model()
    if model is None:
        raise RuntimeError("too few measurements to fit the GP yet")
    with torch.no_grad():
        post = model.posterior(X.double())
        return post.mean, post.variance.clamp_min(0).sqrt()
