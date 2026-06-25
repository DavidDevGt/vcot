"""Tests del medidor de coste REAL de contenedor (vcot.telemetry.billing)."""

import pytest

from vcot.telemetry import ContainerCost, ContainerMeter, projected_container_cost
from vcot.telemetry.rates import (
    CPU_PER_SECOND,
    MEM_PER_SECOND_PER_GIB,
    MIN_CONTAINER_CORES,
    gpu_rate,
)


def _seq(*values):
    """Reader/clock determinista: devuelve los valores dados en orden."""
    it = iter(values)
    return lambda: next(it)


def test_billed_lifetime_is_full_container_window():
    # clock: t0=0 (init), t_stop=100 → 100 s de vida facturada.
    meter = ContainerMeter(
        "A100-80GB",
        clock=_seq(0.0, 100.0),
        cpu_reader=_seq(10.0, 25.0),  # 15 núcleo-s usados
        mem_reader=_seq(8.0),         # 8 GiB pico
    )
    cost = meter.stop()

    assert cost.billed_s == pytest.approx(100.0)
    # GPU = vida completa × tarifa (no solo la inferencia).
    assert cost.gpu_cost_usd == pytest.approx(100.0 * gpu_rate("A100-80GB"))
    # CPU = núcleo-segundos usados (> mínimo reservado) × tarifa.
    assert cost.cpu_core_s == pytest.approx(15.0)
    assert cost.cpu_cost_usd == pytest.approx(15.0 * CPU_PER_SECOND)
    # Memoria = pico × vida × tarifa.
    assert cost.mem_gib == pytest.approx(8.0)
    assert cost.mem_cost_usd == pytest.approx(8.0 * 100.0 * MEM_PER_SECOND_PER_GIB)
    assert cost.real_cost_usd == pytest.approx(
        cost.gpu_cost_usd + cost.cpu_cost_usd + cost.mem_cost_usd
    )
    assert cost.measured is True


def test_cpu_floored_to_min_reserved_cores():
    # Uso de CPU por debajo del mínimo reservado → se factura el mínimo (Modal cobra max).
    meter = ContainerMeter(
        "L4",
        clock=_seq(0.0, 80.0),
        cpu_reader=_seq(0.0, 1.0),  # solo 1 núcleo-s usado en 80 s
        mem_reader=_seq(2.0),
    )
    cost = meter.stop()
    assert cost.cpu_core_s == pytest.approx(MIN_CONTAINER_CORES * 80.0)


def test_falls_back_to_estimate_off_linux():
    # Sin cgroup (None): CPU cae al mínimo, memoria usa el fallback, measured=False.
    meter = ContainerMeter(
        "T4",
        mem_gib_fallback=4.0,
        clock=_seq(0.0, 50.0),
        cpu_reader=_seq(None, None),
        mem_reader=_seq(None),
    )
    cost = meter.stop()
    assert cost.measured is False
    assert cost.cpu_core_s == pytest.approx(MIN_CONTAINER_CORES * 50.0)
    assert cost.mem_gib == pytest.approx(4.0)
    assert cost.mem_cost_usd == pytest.approx(4.0 * 50.0 * MEM_PER_SECOND_PER_GIB)


def test_no_gpu_container_costs_cpu_and_mem_only():
    meter = ContainerMeter(
        None,
        clock=_seq(0.0, 30.0),
        cpu_reader=_seq(0.0, 100.0),
        mem_reader=_seq(1.0),
    )
    cost = meter.stop()
    assert cost.gpu_cost_usd == 0.0
    assert cost.real_cost_usd > 0.0  # CPU + memoria igual cuestan


def test_projected_includes_load_and_idle_tail():
    # La proyección del cliente: vida ≈ carga + activo + idle del scaledown.
    proj = projected_container_cost(
        gpu="A100-80GB",
        active_s=12.0,
        model_load_s=40.0,
        scaledown_window=120.0,
    )
    assert proj.billed_s == pytest.approx(172.0)
    assert proj.gpu_cost_usd == pytest.approx(172.0 * gpu_rate("A100-80GB"))
    assert proj.measured is False


def test_projected_dwarfs_marginal_for_sparse_calls():
    # El coste real estimado de una llamada suelta es muchísimo mayor que el marginal.
    active_s = 12.0
    marginal = active_s * gpu_rate("A100-80GB")
    proj = projected_container_cost(
        gpu="A100-80GB",
        active_s=active_s,
        model_load_s=40.0,
        scaledown_window=120.0,
    )
    assert proj.real_cost_usd > 10 * marginal


def test_as_dict_shape():
    cost = projected_container_cost(gpu="L4", active_s=1.0, scaledown_window=2.0)
    d = cost.as_dict()
    assert set(d) == {
        "gpu", "billed_s", "gpu_cost_usd", "cpu_core_s", "cpu_cost_usd",
        "mem_gib", "mem_cost_usd", "real_cost_usd", "measured",
    }
    assert isinstance(cost, ContainerCost)
