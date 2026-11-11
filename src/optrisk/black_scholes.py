"""Closed-form Black-Scholes price and Greeks for European options without dividends.

All functions are pure and broadcast over their numeric inputs. Vega is per 1.00 of
volatility and theta is per year.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

OptionKind = Literal["call", "put"]
FloatArray = NDArray[np.float64]

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


@dataclass(frozen=True, slots=True)
class Greeks:
    price: FloatArray
    delta: FloatArray
    gamma: FloatArray
    vega: FloatArray
    theta: FloatArray


def check_kind(kind: str) -> OptionKind:
    """Return ``kind`` narrowed to ``OptionKind``, or raise ``ValueError``."""
    if kind == "call":
        return "call"
    if kind == "put":
        return "put"
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def validate_inputs(
    spot: ArrayLike, strike: ArrayLike, expiry: ArrayLike, rate: ArrayLike, vol: ArrayLike
) -> tuple[FloatArray, ...]:
    """Convert the inputs to float arrays and raise ``ValueError`` if any is out of range."""
    s, k, t, r, v = (np.asarray(x, dtype=np.float64) for x in (spot, strike, expiry, rate, vol))
    for name, values in (("spot", s), ("strike", k), ("expiry", t), ("rate", r), ("vol", v)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
    if not np.all(s > 0):
        raise ValueError("spot must be positive")
    if not np.all(k > 0):
        raise ValueError("strike must be positive")
    if not np.all(t >= 0):
        raise ValueError("expiry must not be negative")
    if not np.all(v >= 0):
        raise ValueError("vol must not be negative")
    return s, k, t, r, v


def greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    expiry: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    kind: OptionKind,
) -> Greeks:
    """Price and all Greeks in one pass; d1 and d2 are computed once."""
    sign = 1.0 if check_kind(kind) == "call" else -1.0
    s, k, t, r, v = validate_inputs(spot, strike, expiry, rate, vol)

    # Where expiry or vol is zero the formulas divide by zero, so evaluate them on
    # harmless stand-in values there and overwrite the results with the limits below.
    degenerate = (t == 0) | (v == 0)
    safe_v = np.where(degenerate, 1.0, v)
    sqrt_t = np.sqrt(np.where(degenerate, 1.0, t))
    discounted_strike = k * np.exp(-r * t)

    d1 = (np.log(s / k) + (r + 0.5 * safe_v**2) * sqrt_t**2) / (safe_v * sqrt_t)
    d2 = d1 - safe_v * sqrt_t
    pdf_d1 = _INV_SQRT_2PI * np.exp(-0.5 * d1**2)
    # For a put, sign = -1 turns N(d) into N(-d), giving both columns of the formula table.
    cdf_d1: FloatArray = ndtr(sign * d1)
    cdf_d2: FloatArray = ndtr(sign * d2)

    price_ = sign * (s * cdf_d1 - discounted_strike * cdf_d2)
    delta_ = sign * cdf_d1
    gamma_ = pdf_d1 / (s * safe_v * sqrt_t)
    vega_ = s * pdf_d1 * sqrt_t
    theta_ = -s * pdf_d1 * safe_v / (2.0 * sqrt_t) - sign * r * discounted_strike * cdf_d2

    moneyness = sign * (s - discounted_strike)
    return Greeks(
        price=np.asarray(np.where(degenerate, np.maximum(moneyness, 0.0), price_)),
        delta=np.asarray(np.where(degenerate, np.where(moneyness > 0, sign, 0.0), delta_)),
        gamma=np.asarray(np.where(degenerate, 0.0, gamma_)),
        vega=np.asarray(np.where(degenerate, 0.0, vega_)),
        theta=np.asarray(np.where(degenerate, 0.0, theta_)),
    )


def price(
    spot: ArrayLike,
    strike: ArrayLike,
    expiry: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    kind: OptionKind,
) -> FloatArray:
    """Option price."""
    return greeks(spot, strike, expiry, rate, vol, kind).price


def delta(
    spot: ArrayLike,
    strike: ArrayLike,
    expiry: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    kind: OptionKind,
) -> FloatArray:
    """Sensitivity of the price to spot."""
    return greeks(spot, strike, expiry, rate, vol, kind).delta


def gamma(
    spot: ArrayLike,
    strike: ArrayLike,
    expiry: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    kind: OptionKind,
) -> FloatArray:
    """Sensitivity of delta to spot."""
    return greeks(spot, strike, expiry, rate, vol, kind).gamma


def vega(
    spot: ArrayLike,
    strike: ArrayLike,
    expiry: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    kind: OptionKind,
) -> FloatArray:
    """Sensitivity of the price to volatility, per 1.00 of vol."""
    return greeks(spot, strike, expiry, rate, vol, kind).vega


def theta(
    spot: ArrayLike,
    strike: ArrayLike,
    expiry: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    kind: OptionKind,
) -> FloatArray:
    """Sensitivity of the price to the passage of time, per year."""
    return greeks(spot, strike, expiry, rate, vol, kind).theta
