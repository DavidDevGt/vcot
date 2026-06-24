import pytest

from vcot.cli import estimate_cost, main
from vcot.telemetry.rates import gpu_rate


def test_estimate_cost_single():
    assert estimate_cost("H100", 6.0) == pytest.approx(gpu_rate("H100") * 6.0)


def test_estimate_cost_scales_with_samples():
    one = estimate_cost("A100-80GB", 10.0, 1)
    thousand = estimate_cost("A100-80GB", 10.0, 1000)
    assert thousand == pytest.approx(one * 1000)


def test_main_prints_estimate(capsys):
    main(["--gpu", "H100", "--compute-s", "6", "--samples", "1000"])
    out = capsys.readouterr().out
    assert "H100" in out
    assert "1000 muestras" in out


def test_main_defaults(capsys):
    main([])
    out = capsys.readouterr().out
    assert "H100" in out  # GPU por defecto
