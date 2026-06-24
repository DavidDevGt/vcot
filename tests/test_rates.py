import math

import pytest

from vcot.telemetry import rates


def test_gpu_rate_known():
    assert rates.gpu_rate("H100") == pytest.approx(0.001097)


def test_gpu_rate_hour_is_derived():
    assert rates.gpu_rate("H100", per="hour") == pytest.approx(0.001097 * 3600)
    # ...y coincide con la columna $/hora del catálogo (IDEA.md §8.1).
    assert rates.gpu_rate("H100", per="hour") == pytest.approx(3.95, abs=0.01)


def test_gpu_rate_case_and_alias_insensitive():
    assert rates.gpu_rate("a100-80gb") == rates.gpu_rate("A100-80GB")
    assert rates.gpu_rate("A100") == rates.gpu_rate("A100-80GB")
    assert rates.gpu_rate("A100-40") == rates.gpu_rate("A100-40GB")


def test_gpu_rate_unknown_raises():
    with pytest.raises(KeyError):
        rates.gpu_rate("RTX-9090")


def test_resource_rate_sums_components():
    expected = (
        rates.GPU_PER_SECOND["H100"]
        + rates.CPU_PER_SECOND * 2
        + rates.MEM_PER_SECOND_PER_GIB * 16
    )
    assert rates.resource_rate("H100", cores=2, mem_gib=16) == pytest.approx(expected)


def test_resource_rate_applies_min_cores():
    # 0.01 núcleos solicitados se facturan al mínimo del contenedor.
    rate = rates.resource_rate(cores=0.01)
    assert rate == pytest.approx(rates.CPU_PER_SECOND * rates.MIN_CONTAINER_CORES)


def test_resource_rate_invalid_per():
    with pytest.raises(ValueError):
        rates.resource_rate("H100", per="minute")


def test_per_second_rates_match_hourly_catalog():
    # Sanity-check de que la tabla no se desincronizó (margen por redondeo del doc).
    catalog_hourly = {"B200": 6.25, "H100": 3.95, "A100-80GB": 2.50, "T4": 0.59}
    for gpu, hourly in catalog_hourly.items():
        assert math.isclose(rates.gpu_rate(gpu, per="hour"), hourly, abs_tol=0.01)
