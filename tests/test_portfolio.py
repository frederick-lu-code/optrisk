from pathlib import Path

import pytest

from optrisk import Position, greeks, load_positions, risk_report

EXAMPLE = Path(__file__).parents[1] / "examples" / "positions.csv"
HEADER = "underlying,kind,strike,expiry,quantity,spot,vol\n"


def write(tmp_path, text):
    path = tmp_path / "positions.csv"
    path.write_text(text)
    return path


def test_load_example():
    positions = load_positions(EXAMPLE)
    assert len(positions) == 3
    assert positions[1] == Position("ABC", "put", 95.0, 0.5, -5.0, 100.0, 0.25)


def test_columns_may_be_in_any_order(tmp_path):
    path = write(
        tmp_path, "vol,spot,quantity,expiry,strike,kind,underlying\n0.2,100,10,1,100,call,ABC\n"
    )
    assert load_positions(path) == [Position("ABC", "call", 100.0, 1.0, 10.0, 100.0, 0.2)]


def test_header_only_file_has_no_positions(tmp_path):
    assert load_positions(write(tmp_path, HEADER)) == []


def test_bad_kind_names_the_line(tmp_path):
    path = write(tmp_path, HEADER + "ABC,call,100,1,10,100,0.2\nABC,straddle,100,1,10,100,0.2\n")
    with pytest.raises(ValueError, match=r"line 3: .*straddle"):
        load_positions(path)


def test_non_numeric_field_names_the_line(tmp_path):
    path = write(tmp_path, HEADER + "ABC,call,one hundred,1,10,100,0.2\n")
    with pytest.raises(ValueError, match="line 2: strike is not a number"):
        load_positions(path)


def test_missing_header_column_names_line_one(tmp_path):
    path = write(tmp_path, "underlying,kind,strike,expiry,quantity,spot\nABC,call,100,1,10,100\n")
    with pytest.raises(ValueError, match="line 1: missing column 'vol'"):
        load_positions(path)


def test_short_row_names_the_missing_column(tmp_path):
    path = write(tmp_path, HEADER + "ABC,call,100,1,10,100\n")
    with pytest.raises(ValueError, match="line 2: missing column 'vol'"):
        load_positions(path)


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "line 1: missing header"),
        (HEADER.strip() + ",note\nABC,call,100,1,10,100,0.2,x\n", "line 1: header must be exactly"),
        (HEADER + "ABC,call,100,1,10,100,0.2,extra\n", "line 2: more fields"),
        (HEADER + ",call,100,1,10,100,0.2\n", "line 2: underlying is empty"),
    ],
)
def test_other_malformed_files_raise(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_positions(write(tmp_path, text))


def test_risk_report_matches_worked_example():
    abc, xyz = risk_report(load_positions(EXAMPLE), 0.05)
    assert (abc.underlying, xyz.underlying) == ("ABC", "XYZ")
    assert (round(abc.market_value, 2), round(xyz.market_value, 2)) == (85.85, 45.46)
    assert round(abc.market_value + xyz.market_value, 2) == 131.31
    abc_greeks = [round(x, 4) for x in (abc.delta, abc.gamma, abc.vega, abc.theta)]
    xyz_greeks = [round(x, 4) for x in (xyz.delta, xyz.gamma, xyz.vega, xyz.theta)]
    assert abc_greeks == [7.8760, 0.0890, 2.5203, -0.1145]
    assert xyz_greeks == [9.0939, 1.1010, 1.9026, -0.3663]


def test_risk_report_sorts_by_underlying_and_rescales_vega_and_theta():
    positions = [
        Position("ZZZ", "put", 100.0, 1.0, -2.0, 100.0, 0.2),
        Position("AAA", "call", 100.0, 1.0, 3.0, 100.0, 0.2),
    ]
    aaa, zzz = risk_report(positions, 0.05)
    assert (aaa.underlying, zzz.underlying) == ("AAA", "ZZZ")
    call = greeks(100.0, 100.0, 1.0, 0.05, 0.2, "call")
    put = greeks(100.0, 100.0, 1.0, 0.05, 0.2, "put")
    assert aaa.market_value == pytest.approx(3 * float(call.price))
    assert aaa.vega == pytest.approx(3 * float(call.vega) / 100)
    assert aaa.theta == pytest.approx(3 * float(call.theta) / 365)
    assert zzz.delta == pytest.approx(-2 * float(put.delta))
    assert zzz.gamma == pytest.approx(-2 * float(put.gamma))


def test_long_and_short_of_the_same_option_net_to_zero():
    positions = [
        Position("ABC", "call", 100.0, 1.0, 5.0, 100.0, 0.2),
        Position("ABC", "call", 100.0, 1.0, -5.0, 100.0, 0.2),
    ]
    (row,) = risk_report(positions, 0.05)
    assert (row.market_value, row.delta, row.gamma, row.vega, row.theta) == (0, 0, 0, 0, 0)


def test_empty_portfolio_gives_empty_report():
    assert risk_report([], 0.05) == []


def test_invalid_position_raises():
    with pytest.raises(ValueError, match="strike"):
        risk_report([Position("ABC", "call", -100.0, 1.0, 1.0, 100.0, 0.2)], 0.05)
