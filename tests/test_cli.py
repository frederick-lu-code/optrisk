from pathlib import Path

import pytest

from optrisk.cli import main

EXAMPLE = Path(__file__).parents[1] / "examples" / "positions.csv"
CONTRACT = ["--spot", "100", "--strike", "100", "--expiry", "1", "--rate", "0.05"]

PRICE_OUTPUT = """\
price   10.450584
delta    0.636831
gamma    0.018762
vega    37.524035
theta   -6.414028
"""

REPORT_OUTPUT = """\
underlying    value     delta     gamma      vega     theta
ABC           85.85    7.8760    0.0890    2.5203   -0.1145
XYZ           45.46    9.0939    1.1010    1.9026   -0.3663
TOTAL        131.31
"""


def test_price(capsys):
    assert main(["price", *CONTRACT, "--vol", "0.2", "--kind", "call"]) == 0
    assert capsys.readouterr().out == PRICE_OUTPUT


def test_iv(capsys):
    assert main(["iv", "--price", "10.450584", *CONTRACT, "--kind", "call"]) == 0
    assert capsys.readouterr().out == "0.200000\n"


def test_report(capsys):
    assert main(["report", str(EXAMPLE), "--rate", "0.05"]) == 0
    assert capsys.readouterr().out == REPORT_OUTPUT


def test_report_columns_stay_aligned_for_long_names(tmp_path, capsys):
    path = tmp_path / "positions.csv"
    path.write_text(
        "underlying,kind,strike,expiry,quantity,spot,vol\n"
        "A_VERY_LONG_UNDERLYING,call,100,1,10,100,0.2\n"
        "B,put,100,1,-10,100,0.2\n"
    )
    assert main(["report", str(path), "--rate", "0.05"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len({len(line) for line in lines[:3]}) == 1
    assert lines[3].startswith("TOTAL")


@pytest.mark.parametrize(
    "argv",
    [
        ["price", *CONTRACT, "--vol", "-0.2", "--kind", "call"],
        ["price", "--spot", "0", "--strike", "100", "--expiry", "1", "--rate", "0.05"]
        + ["--vol", "0.2", "--kind", "put"],
        ["iv", "--price", "150", *CONTRACT, "--kind", "call"],
        ["iv", "--price", "10", "--spot", "100", "--strike", "100", "--expiry", "0"]
        + ["--rate", "0.05", "--kind", "call"],
        ["report", "no-such-file.csv", "--rate", "0.05"],
    ],
)
def test_bad_input_prints_error_and_exits_2(argv, capsys):
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err


def test_bad_csv_reports_the_line(tmp_path, capsys):
    path = tmp_path / "positions.csv"
    path.write_text("underlying,kind,strike,expiry,quantity,spot,vol\nABC,swap,100,1,10,100,0.2\n")
    assert main(["report", str(path), "--rate", "0.05"]) == 2
    assert capsys.readouterr().err.startswith("error: line 2: ")


@pytest.mark.parametrize(
    "argv",
    [[], ["price", "--spot", "100"], ["price", *CONTRACT, "--vol", "0.2", "--kind", "swap"]],
)
def test_usage_errors_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
