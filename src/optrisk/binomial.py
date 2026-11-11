"""Cox-Ross-Rubinstein binomial tree for European options.

An independent numerical cross-check on the closed-form prices in ``black_scholes``.
"""

import math

import numpy as np

from optrisk.black_scholes import OptionKind, check_kind, validate_inputs


def binomial_price(
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    vol: float,
    kind: OptionKind,
    *,
    steps: int = 500,
) -> float:
    """Price a European option on a CRR tree with ``steps`` time steps."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    sign = 1.0 if check_kind(kind) == "call" else -1.0
    validate_inputs(spot, strike, expiry, rate, vol)
    if expiry == 0 or vol == 0:
        # The tree collapses to a single path (u == d): discounted intrinsic value.
        return max(sign * (spot - strike * math.exp(-rate * expiry)), 0.0)

    dt = expiry / steps
    up = math.exp(vol * math.sqrt(dt))
    down = 1.0 / up
    prob_up = (math.exp(rate * dt) - down) / (up - down)
    if not 0.0 <= prob_up <= 1.0:
        raise ValueError(
            f"risk-neutral probability {prob_up:g} is outside [0, 1]; "
            "increase steps so that |rate| * sqrt(expiry / steps) <= vol"
        )
    disc = math.exp(-rate * dt)

    # Node j at expiry has j up-moves and (steps - j) down-moves.
    up_moves = np.arange(steps + 1)
    terminal_spot = spot * up ** (2.0 * up_moves - steps)
    values = np.maximum(sign * (terminal_spot - strike), 0.0)
    for _ in range(steps):
        values = disc * (prob_up * values[1:] + (1.0 - prob_up) * values[:-1])
    return float(values[0])
