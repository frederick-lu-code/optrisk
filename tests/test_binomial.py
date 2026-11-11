import math

import pytest

from optrisk import binomial_price, price

REF = (100.0, 100.0, 1.0, 0.05, 0.2)


@pytest.mark.parametrize("steps, expected", [(500, 10.446585), (1000, 10.448584)])
def test_reference_values(steps, expected):
    assert binomial_price(*REF, "call", steps=steps) == pytest.approx(expected, abs=1e-6)


def test_default_is_500_steps():
    assert binomial_price(*REF, "call") == binomial_price(*REF, "call", steps=500)


@pytest.mark.parametrize(
    "spot, strike, expiry, rate, vol, kind",
    [
        (100.0, 100.0, 1.0, 0.05, 0.2, "call"),
        (100.0, 100.0, 1.0, 0.05, 0.2, "put"),
        (120.0, 100.0, 0.5, 0.03, 0.35, "call"),  # in the money
        (80.0, 100.0, 2.0, 0.01, 0.5, "call"),  # out of the money
        (80.0, 100.0, 2.0, 0.01, 0.5, "put"),  # in the money
        (120.0, 100.0, 0.5, 0.03, 0.35, "put"),  # out of the money
    ],
)
def test_converges_to_closed_form(spot, strike, expiry, rate, vol, kind):
    tree = binomial_price(spot, strike, expiry, rate, vol, kind, steps=1000)
    assert abs(tree - float(price(spot, strike, expiry, rate, vol, kind))) < 0.01


def test_tree_satisfies_put_call_parity():
    call = binomial_price(*REF, "call", steps=200)
    put = binomial_price(*REF, "put", steps=200)
    assert call - put == pytest.approx(100.0 - 100.0 * math.exp(-0.05), abs=1e-9)


def test_one_step_tree_matches_hand_calculation():
    up, rate = math.exp(0.2), 0.05
    p = (math.exp(rate) - 1 / up) / (up - 1 / up)
    expected = math.exp(-rate) * p * (100.0 * up - 100.0)
    assert binomial_price(*REF, "call", steps=1) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("expiry, vol", [(0.0, 0.2), (1.0, 0.0)])
def test_zero_expiry_or_vol_matches_closed_form(expiry, vol):
    for kind in ("call", "put"):
        tree = binomial_price(110.0, 100.0, expiry, 0.05, vol, kind)
        assert tree == pytest.approx(float(price(110.0, 100.0, expiry, 0.05, vol, kind)))


@pytest.mark.parametrize("steps", [0, -3])
def test_too_few_steps_raises(steps):
    with pytest.raises(ValueError, match="steps"):
        binomial_price(*REF, "call", steps=steps)


@pytest.mark.parametrize(
    "spot, strike, expiry, vol, kind",
    [
        (-100.0, 100.0, 1.0, 0.2, "call"),
        (100.0, 0.0, 1.0, 0.2, "call"),
        (100.0, 100.0, -1.0, 0.2, "put"),
        (100.0, 100.0, 1.0, -0.2, "put"),
        (100.0, 100.0, 1.0, 0.2, "digital"),
    ],
)
def test_invalid_inputs_raise(spot, strike, expiry, vol, kind):
    with pytest.raises(ValueError):
        binomial_price(spot, strike, expiry, 0.05, vol, kind)


def test_step_too_coarse_for_a_valid_probability_raises():
    with pytest.raises(ValueError, match="increase steps"):
        binomial_price(100.0, 100.0, 1.0, 0.5, 0.01, "call", steps=1)
