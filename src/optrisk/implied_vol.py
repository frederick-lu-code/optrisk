"""Implied volatility by Newton's method, safeguarded by a bisection bracket."""

import math

from optrisk.black_scholes import OptionKind, greeks
from optrisk.black_scholes import price as bs_price

_VOL_LO = 1e-6
_VOL_HI = 5.0
_INITIAL_GUESS = 0.2
_MIN_VEGA = 1e-12


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    kind: OptionKind,
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Volatility at which the Black-Scholes price equals ``price`` to within ``tol``.

    Raises ``ValueError`` for invalid inputs or a price outside the no-arbitrage bounds,
    and ``RuntimeError`` if the solver does not converge in ``max_iter`` iterations.
    """
    if not expiry > 0:
        raise ValueError("expiry must be positive")
    if not math.isfinite(price):
        raise ValueError("price must be finite")
    # The zero-vol price is the lower no-arbitrage bound; this call also validates the inputs.
    lower = float(bs_price(spot, strike, expiry, rate, 0.0, kind))
    if kind == "call":
        upper, upper_name = spot, "spot"
    else:
        upper, upper_name = strike * math.exp(-rate * expiry), "the discounted strike"
    if price < lower:
        raise ValueError(
            f"price {price:g} is below the lower bound {lower:g} (discounted intrinsic value)"
        )
    if price >= upper:
        raise ValueError(f"price {price:g} is at or above the upper bound {upper:g} ({upper_name})")
    if price - lower < tol:
        return 0.0

    lo, hi = _VOL_LO, _VOL_HI
    if float(bs_price(spot, strike, expiry, rate, hi, kind)) < price:
        raise ValueError(f"price {price:g} implies a volatility above {hi:g}")

    sigma = _INITIAL_GUESS
    for _ in range(max_iter):
        at_sigma = greeks(spot, strike, expiry, rate, sigma, kind)
        diff = float(at_sigma.price) - price
        if abs(diff) < tol:
            return sigma
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        slope = float(at_sigma.vega)
        newton = sigma - diff / slope if slope >= _MIN_VEGA else math.nan
        sigma = newton if lo < newton < hi else 0.5 * (lo + hi)
    raise RuntimeError(f"implied volatility did not converge in {max_iter} iterations")
