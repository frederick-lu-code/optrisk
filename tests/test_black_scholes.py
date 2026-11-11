import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optrisk import delta, gamma, greeks, price, theta, vega

REF = (100.0, 100.0, 1.0, 0.05, 0.2)
REFERENCE = {
    "call": (10.450584, 0.636831, 0.018762, 37.524035, -6.414028),
    "put": (5.573526, -0.363169, 0.018762, 37.524035, -1.657880),
}

# (spot, strike, expiry, rate, vol): at, in and out of the money, short and long dated.
CASES = [
    (100.0, 100.0, 1.0, 0.05, 0.2),
    (120.0, 100.0, 0.5, 0.03, 0.35),
    (80.0, 100.0, 2.0, 0.01, 0.5),
    (48.0, 50.0, 0.25, 0.05, 0.3),
]

spots = st.floats(10, 500)
strikes = st.floats(10, 500)
expiries = st.floats(0.05, 3)
rates = st.floats(0, 0.1)
vols = st.floats(0.05, 1.0)
market = given(spots, strikes, expiries, rates, vols)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_reference_values(kind):
    expected = REFERENCE[kind]
    functions = (price, delta, gamma, vega, theta)
    for function, value in zip(functions, expected, strict=True):
        assert float(function(*REF, kind)) == pytest.approx(value, abs=1e-6)
    g = greeks(*REF, kind)
    assert (float(g.price), float(g.delta), float(g.gamma), float(g.vega), float(g.theta)) == (
        pytest.approx(expected, abs=1e-6)
    )


def test_scalar_inputs_return_zero_dimensional_arrays():
    result = price(*REF, "call")
    assert isinstance(result, np.ndarray)
    assert result.shape == ()


def test_strike_vector_broadcasts_against_scalar_spot():
    strike = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    result = price(100.0, strike, 1.0, 0.05, 0.2, "call")
    assert result.shape == (5,)
    for element, k in zip(result, strike, strict=True):
        assert element == pytest.approx(float(price(100.0, k, 1.0, 0.05, 0.2, "call")), abs=1e-12)


def test_two_dimensional_broadcast():
    spot = np.array([[90.0], [100.0], [110.0]])
    strike = np.array([95.0, 105.0])
    assert greeks(spot, strike, 1.0, 0.05, 0.2, "put").theta.shape == (3, 2)


@pytest.mark.parametrize("expiry, vol", [(0.0, 0.2), (1.0, 0.0), (0.0, 0.0)])
@pytest.mark.parametrize("spot", [90.0, 110.0])
def test_zero_expiry_or_vol_gives_discounted_intrinsic(spot, expiry, vol):
    strike, rate = 100.0, 0.05
    forward_gap = spot - strike * math.exp(-rate * expiry)
    call = greeks(spot, strike, expiry, rate, vol, "call")
    put = greeks(spot, strike, expiry, rate, vol, "put")
    assert float(call.price) == pytest.approx(max(forward_gap, 0.0), abs=1e-12)
    assert float(put.price) == pytest.approx(max(-forward_gap, 0.0), abs=1e-12)
    assert float(call.delta) == (1.0 if forward_gap > 0 else 0.0)
    assert float(put.delta) == (-1.0 if forward_gap < 0 else 0.0)
    for g in (call, put):
        assert float(g.gamma) == 0.0
        assert float(g.vega) == 0.0
        assert float(g.theta) == 0.0


def test_edge_cases_mix_with_regular_elements_in_one_array():
    result = price(100.0, 100.0, np.array([0.0, 1.0]), 0.05, np.array([0.2, 0.2]), "call")
    assert result[0] == 0.0
    assert result[1] == pytest.approx(10.450584, abs=1e-6)


@pytest.mark.parametrize(
    "spot, strike, expiry, rate, vol, kind",
    [
        (0.0, 100.0, 1.0, 0.05, 0.2, "call"),
        (-1.0, 100.0, 1.0, 0.05, 0.2, "call"),
        (100.0, 0.0, 1.0, 0.05, 0.2, "put"),
        (100.0, 100.0, -0.1, 0.05, 0.2, "call"),
        (100.0, 100.0, 1.0, 0.05, -0.2, "put"),
        (math.nan, 100.0, 1.0, 0.05, 0.2, "call"),
        (100.0, 100.0, 1.0, math.inf, 0.2, "call"),
        (100.0, 100.0, 1.0, 0.05, 0.2, "straddle"),
    ],
)
def test_invalid_inputs_raise(spot, strike, expiry, rate, vol, kind):
    with pytest.raises(ValueError):
        price(spot, strike, expiry, rate, vol, kind)


def test_one_bad_element_in_an_array_raises():
    with pytest.raises(ValueError, match="strike"):
        price(100.0, np.array([100.0, -5.0]), 1.0, 0.05, 0.2, "call")


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("spot, strike, expiry, rate, vol", CASES)
def test_greeks_match_central_finite_differences(spot, strike, expiry, rate, vol, kind):
    ds, dv, dt = 1e-4 * spot, 1e-5, 1e-5

    def p(s=spot, t=expiry, v=vol):
        return float(price(s, strike, t, rate, v, kind))

    def d(s):
        return float(delta(s, strike, expiry, rate, vol, kind))

    g = greeks(spot, strike, expiry, rate, vol, kind)
    assert float(g.delta) == pytest.approx((p(s=spot + ds) - p(s=spot - ds)) / (2 * ds), rel=1e-4)
    assert float(g.vega) == pytest.approx((p(v=vol + dv) - p(v=vol - dv)) / (2 * dv), rel=1e-4)
    assert float(g.theta) == pytest.approx(
        -(p(t=expiry + dt) - p(t=expiry - dt)) / (2 * dt), rel=1e-4
    )
    assert float(g.gamma) == pytest.approx((d(spot + ds) - d(spot - ds)) / (2 * ds), rel=1e-4)


@market
@settings(deadline=None)
def test_put_call_parity(spot, strike, expiry, rate, vol):
    call = float(price(spot, strike, expiry, rate, vol, "call"))
    put = float(price(spot, strike, expiry, rate, vol, "put"))
    forward_gap = spot - strike * math.exp(-rate * expiry)
    assert (call - put) / spot == pytest.approx(forward_gap / spot, abs=1e-8)


@market
@settings(deadline=None)
def test_no_arbitrage_bounds(spot, strike, expiry, rate, vol):
    discounted_strike = strike * math.exp(-rate * expiry)
    slack = 1e-9 * spot  # floating-point rounding only
    call = float(price(spot, strike, expiry, rate, vol, "call"))
    put = float(price(spot, strike, expiry, rate, vol, "put"))
    assert max(spot - discounted_strike, 0.0) - slack <= call <= spot + slack
    assert max(discounted_strike - spot, 0.0) - slack <= put <= discounted_strike + slack


@given(spots, strikes, expiries, rates, vols, vols, st.sampled_from(["call", "put"]))
@settings(deadline=None)
def test_price_is_non_decreasing_in_vol(spot, strike, expiry, rate, vol_a, vol_b, kind):
    low, high = sorted((vol_a, vol_b))
    prices = price(spot, strike, expiry, rate, np.array([low, high]), kind)
    assert prices[1] >= prices[0] - 1e-9 * spot


@given(spots, strikes, strikes, expiries, rates, vols)
@settings(deadline=None)
def test_call_price_is_non_increasing_in_strike(spot, strike_a, strike_b, expiry, rate, vol):
    low, high = sorted((strike_a, strike_b))
    prices = price(spot, np.array([low, high]), expiry, rate, vol, "call")
    assert prices[1] <= prices[0] + 1e-9 * spot


@market
@settings(deadline=None)
def test_greek_signs(spot, strike, expiry, rate, vol):
    call = greeks(spot, strike, expiry, rate, vol, "call")
    put = greeks(spot, strike, expiry, rate, vol, "put")
    assert 0.0 <= float(call.delta) <= 1.0
    assert -1.0 <= float(put.delta) <= 0.0
    for g in (call, put):
        assert float(g.gamma) >= 0.0
        assert float(g.vega) >= 0.0
