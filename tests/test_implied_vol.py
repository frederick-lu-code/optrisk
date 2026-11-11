import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from optrisk import implied_vol, price, vega


@pytest.mark.parametrize("kind, target", [("call", 10.450584), ("put", 5.573526)])
def test_reference_price_recovers_reference_vol(kind, target):
    assert implied_vol(target, 100.0, 100.0, 1.0, 0.05, kind) == pytest.approx(0.2, abs=1e-6)


@pytest.mark.parametrize("vol", [0.05, 0.8, 2.5, 4.5])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_recovers_low_and_high_vols(vol, kind):
    # Far from the 0.2 starting guess, so the bisection safeguard is exercised.
    target = float(price(100.0, 110.0, 0.5, 0.02, vol, kind))
    assert implied_vol(target, 100.0, 110.0, 0.5, 0.02, kind) == pytest.approx(vol, abs=1e-6)


def test_time_value_below_tolerance_returns_zero():
    # Out of the money at 1% vol the option is worth less than tol, so there is nothing to solve.
    target = float(price(100.0, 110.0, 0.5, 0.02, 0.01, "call"))
    assert 0.0 < target < 1e-8
    assert implied_vol(target, 100.0, 110.0, 0.5, 0.02, "call") == 0.0


def test_price_at_lower_bound_returns_zero():
    intrinsic = 120.0 - 100.0 * math.exp(-0.05)
    assert implied_vol(intrinsic, 120.0, 100.0, 1.0, 0.05, "call") == 0.0
    assert implied_vol(0.0, 80.0, 100.0, 1.0, 0.05, "call") == 0.0


def test_price_below_intrinsic_raises():
    intrinsic = 120.0 - 100.0 * math.exp(-0.05)
    with pytest.raises(ValueError, match="lower bound"):
        implied_vol(intrinsic - 0.5, 120.0, 100.0, 1.0, 0.05, "call")
    with pytest.raises(ValueError, match="lower bound"):
        implied_vol(1.0, 80.0, 100.0, 1.0, 0.05, "put")


@pytest.mark.parametrize("target", [100.0, 150.0])
def test_call_price_at_or_above_spot_raises(target):
    with pytest.raises(ValueError, match="upper bound"):
        implied_vol(target, 100.0, 100.0, 1.0, 0.05, "call")


def test_put_price_at_or_above_discounted_strike_raises():
    with pytest.raises(ValueError, match="upper bound"):
        implied_vol(100.0 * math.exp(-0.05), 100.0, 100.0, 1.0, 0.05, "put")


def test_price_needing_vol_above_bracket_raises():
    target = float(price(100.0, 100.0, 1.0, 0.05, 6.0, "call"))
    with pytest.raises(ValueError, match="above 5"):
        implied_vol(target, 100.0, 100.0, 1.0, 0.05, "call")


@pytest.mark.parametrize(
    "spot, strike, expiry",
    [(-100.0, 100.0, 1.0), (100.0, 0.0, 1.0), (100.0, 100.0, 0.0), (100.0, 100.0, -1.0)],
)
def test_invalid_inputs_raise(spot, strike, expiry):
    with pytest.raises(ValueError):
        implied_vol(5.0, spot, strike, expiry, 0.05, "call")


def test_running_out_of_iterations_raises_runtime_error():
    target = float(price(100.0, 110.0, 0.5, 0.02, 2.5, "call"))
    with pytest.raises(RuntimeError, match="converge"):
        implied_vol(target, 100.0, 110.0, 0.5, 0.02, "call", max_iter=1)


@given(
    st.floats(10, 500),
    st.floats(10, 500),
    st.floats(0.05, 3),
    st.floats(0, 0.1),
    st.floats(0.05, 1.0),
    st.sampled_from(["call", "put"]),
)
@settings(deadline=None)
def test_round_trip(spot, strike, expiry, rate, vol, kind):
    target = float(price(spot, strike, expiry, rate, vol, kind))
    lower_bound = float(price(spot, strike, expiry, rate, 0.0, kind))
    assume(float(vega(spot, strike, expiry, rate, vol, kind)) >= 0.01)
    assume(target - lower_bound >= 1e-6)
    assert implied_vol(target, spot, strike, expiry, rate, kind) == pytest.approx(vol, abs=1e-5)
