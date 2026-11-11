# optrisk

`optrisk` is a small Python library and command-line tool that prices European options with
Black-Scholes, solves for implied volatility, and cross-checks prices on a binomial tree. Given a
CSV of positions it prints value and Greeks aggregated per underlying.

## Install and check

Requires Python 3.12. Runtime dependencies are `numpy` and `scipy` only.

```
pip install -e ".[dev]"
pytest                 # tests (warnings are errors)
ruff check .           # lint
mypy --strict src      # type-check
```

## Usage

```
$ optrisk price --spot 100 --strike 100 --expiry 1 --rate 0.05 --vol 0.2 --kind call
price   10.450584
delta    0.636831
gamma    0.018762
vega    37.524035
theta   -6.414028

$ optrisk iv --price 10.450584 --spot 100 --strike 100 --expiry 1 --rate 0.05 --kind call
0.200000

$ optrisk report examples/positions.csv --rate 0.05
underlying    value     delta     gamma      vega     theta
ABC           85.85    7.8760    0.0890    2.5203   -0.1145
XYZ           45.46    9.0939    1.1010    1.9026   -0.3663
TOTAL        131.31
```

Rates and vols are decimals (`0.05` is 5%) and expiry is in years. `price` and the library report
vega per 1.00 of vol and theta per year; `report` shows vega per vol point and theta per calendar
day. Bad input prints `error: <message>` and exits with status 2. The same functions are importable:
`from optrisk import greeks, implied_vol, binomial_price, load_positions, risk_report`.

## Design decisions

- **Vectorised pricing.** Pricing functions broadcast over NumPy arrays, so a whole book is priced
  in a few array operations and `risk_report` needs one call per underlying and option kind.
- **One implementation of the formulas.** `greeks` computes everything from a single d1/d2.
  `price`, `delta` and the rest just pick a field from it. Tradeoff: asking for one Greek computes
  all five, which costs a little time but leaves no duplicated formula to drift out of sync.
- **Newton with a bisection fallback.** Newton converges in a handful of steps near the money, but
  vega collapses for deep in- or out-of-the-money options and a raw step can overshoot. Keeping a
  bracket and bisecting whenever a step leaves it guarantees convergence.
- **Independent checks.** The binomial tree shares no formula with the closed form, so agreement
  verifies the prices. Finite-difference tests verify each Greek is the derivative of `price`.
  Property-based tests check parity, bounds, monotonicity and the implied-vol round trip.

## Choices where the spec was silent

- At zero expiry or zero vol, "in the money" for delta means spot against the *discounted* strike,
  matching the price. Theta is 0 there as specified, although a zero-vol in-the-money option
  really does accrue interest.
- NaN, infinite and non-`call`/`put` inputs raise `ValueError`, like the specified range checks.
- `implied_vol` returns `0.0` whenever time value is below `tol`, so a nearly worthless
  out-of-the-money option reports zero vol rather than failing.
- `binomial_price` raises `ValueError` if the step is too coarse for a valid risk-neutral
  probability, and returns discounted intrinsic value at zero expiry or zero vol.
- `load_positions` rejects extra columns and empty underlyings, trims whitespace, and treats `kind`
  as case-sensitive. Range errors such as a negative strike surface from `risk_report` without a
  line number.
- The CLI also reports an unreadable file as `error: ...` with status 2. Solver non-convergence
  exits with status 1. `--rate` is always required, with no default.
- `mypy` ignores missing type information for `scipy` only, because `scipy-stubs` is not an allowed
  dependency. There is no CI workflow and no licence file yet.

## Limitations and next steps

No dividends, European exercise only, and one flat rate and vol per option (no curve, smile or term
structure). A next version would add a dividend yield, American exercise on the existing tree, rho
and second-order Greeks, per-position rates, and a vectorised implied-vol solver.
