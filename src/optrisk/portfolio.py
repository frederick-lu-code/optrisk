"""Positions, CSV loading and per-underlying risk aggregation."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from optrisk.black_scholes import OptionKind, check_kind, greeks

COLUMNS = ("underlying", "kind", "strike", "expiry", "quantity", "spot", "vol")
_KINDS: tuple[OptionKind, ...] = ("call", "put")


@dataclass(frozen=True, slots=True)
class Position:
    underlying: str
    kind: OptionKind
    strike: float
    expiry: float  # years
    quantity: float  # signed; negative is short
    spot: float
    vol: float


@dataclass(frozen=True, slots=True)
class UnderlyingRisk:
    underlying: str
    market_value: float
    delta: float
    gamma: float
    vega: float  # per 1 vol point (0.01)
    theta: float  # per calendar day


def _parse_row(row: dict[str | None, str | None], line: int) -> Position:
    if None in row:
        raise ValueError(f"line {line}: more fields than header columns")
    fields: dict[str, str] = {}
    for column in COLUMNS:
        value = row[column]
        if value is None:
            raise ValueError(f"line {line}: missing column '{column}'")
        fields[column] = value.strip()
    if not fields["underlying"]:
        raise ValueError(f"line {line}: underlying is empty")
    try:
        kind = check_kind(fields["kind"])
    except ValueError as exc:
        raise ValueError(f"line {line}: {exc}") from None
    numbers: dict[str, float] = {}
    for column in ("strike", "expiry", "quantity", "spot", "vol"):
        try:
            numbers[column] = float(fields[column])
        except ValueError:
            raise ValueError(f"line {line}: {column} is not a number: {fields[column]!r}") from None
    return Position(underlying=fields["underlying"], kind=kind, **numbers)


def load_positions(path: Path) -> list[Position]:
    """Read positions from a CSV file with a header row naming exactly ``COLUMNS``."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if header is None:
            raise ValueError("line 1: missing header row")
        for column in COLUMNS:
            if column not in header:
                raise ValueError(f"line 1: missing column '{column}'")
        unexpected = [column for column in header if column not in COLUMNS]
        if unexpected or len(header) != len(COLUMNS):
            raise ValueError(f"line 1: header must be exactly the columns {', '.join(COLUMNS)}")
        return [_parse_row(row, reader.line_num) for row in reader]


def risk_report(positions: list[Position], rate: float) -> list[UnderlyingRisk]:
    """Aggregate value and Greeks per underlying, sorted by underlying name.

    Vega is rescaled to 1 vol point and theta to one calendar day.
    """
    by_underlying: dict[str, list[Position]] = {}
    for position in positions:
        by_underlying.setdefault(position.underlying, []).append(position)

    report = []
    for underlying in sorted(by_underlying):
        totals = np.zeros(5)
        for kind in _KINDS:
            legs = [p for p in by_underlying[underlying] if p.kind == kind]
            if not legs:
                continue
            quantity, spot, strike, expiry, vol = np.array(
                [(p.quantity, p.spot, p.strike, p.expiry, p.vol) for p in legs]
            ).T
            g = greeks(spot, strike, expiry, rate, vol, kind)
            totals += np.stack([g.price, g.delta, g.gamma, g.vega, g.theta]) @ quantity
        value, delta, gamma, vega, theta = (float(x) for x in totals)
        report.append(
            UnderlyingRisk(
                underlying=underlying,
                market_value=value,
                delta=delta,
                gamma=gamma,
                vega=vega / 100.0,
                theta=theta / 365.0,
            )
        )
    return report
