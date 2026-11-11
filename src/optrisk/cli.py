"""Command-line entry point: ``optrisk price``, ``optrisk iv`` and ``optrisk report``."""

import argparse
import sys
from pathlib import Path

from optrisk.black_scholes import greeks
from optrisk.implied_vol import implied_vol
from optrisk.portfolio import load_positions, risk_report


def _price_lines(args: argparse.Namespace) -> list[str]:
    g = greeks(args.spot, args.strike, args.expiry, args.rate, args.vol, args.kind)
    names = ("price", "delta", "gamma", "vega", "theta")
    return [f"{name:<5}{float(getattr(g, name)):>12.6f}" for name in names]


def _iv_lines(args: argparse.Namespace) -> list[str]:
    sigma = implied_vol(args.price, args.spot, args.strike, args.expiry, args.rate, args.kind)
    return [f"{sigma:.6f}"]


def _report_lines(args: argparse.Namespace) -> list[str]:
    rows = risk_report(load_positions(args.positions), args.rate)
    width = max([len("underlying")] + [len(row.underlying) for row in rows])
    header = ("value", "delta", "gamma", "vega", "theta")
    lines = [f"{'underlying':<{width}}{header[0]:>9}" + "".join(f"{h:>10}" for h in header[1:])]
    for row in rows:
        risks = (row.delta, row.gamma, row.vega, row.theta)
        lines.append(
            f"{row.underlying:<{width}}{row.market_value:>9.2f}"
            + "".join(f"{x:>10.4f}" for x in risks)
        )
    total = sum(row.market_value for row in rows)
    lines.append(f"{'TOTAL':<{width}}{total:>9.2f}")
    return lines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optrisk", description="Black-Scholes pricing and risk for European options."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_contract(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--spot", type=float, required=True, help="underlying price")
        sub.add_argument("--strike", type=float, required=True)
        sub.add_argument("--expiry", type=float, required=True, help="years to expiry")
        sub.add_argument("--rate", type=float, required=True, help="annual rate, 0.05 = 5%%")

    price = commands.add_parser("price", help="price and Greeks of one option")
    add_contract(price)
    price.add_argument("--vol", type=float, required=True, help="annual vol, 0.2 = 20%%")
    price.add_argument("--kind", choices=("call", "put"), required=True)
    price.set_defaults(handler=_price_lines)

    iv = commands.add_parser("iv", help="implied volatility from an option price")
    iv.add_argument("--price", type=float, required=True, help="option price")
    add_contract(iv)
    iv.add_argument("--kind", choices=("call", "put"), required=True)
    iv.set_defaults(handler=_iv_lines)

    report = commands.add_parser("report", help="risk per underlying for a CSV of positions")
    report.add_argument("positions", type=Path, help="CSV file of positions")
    report.add_argument("--rate", type=float, required=True, help="annual rate, 0.05 = 5%%")
    report.set_defaults(handler=_report_lines)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        lines: list[str] = args.handler(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
