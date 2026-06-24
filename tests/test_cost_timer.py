import time

import pytest

from vcot.telemetry import CostTimer, cost_timer
from vcot.telemetry.rates import gpu_rate


def test_measures_elapsed_and_cost():
    with cost_timer(gpu="H100") as t:
        time.sleep(0.02)
    assert t.seconds >= 0.02
    assert t.cost == pytest.approx(t.seconds * gpu_rate("H100"))


def test_cost_is_zero_before_entering():
    t = CostTimer(gpu="H100")
    assert t.seconds == 0.0
    assert t.cost == 0.0


def test_rate_combines_gpu_cpu_mem():
    from vcot.telemetry.rates import resource_rate

    t = CostTimer(gpu="A100-80GB", cores=4, mem_gib=8)
    assert t.rate_per_second == pytest.approx(
        resource_rate("A100-80GB", cores=4, mem_gib=8)
    )


def test_no_gpu_stage():
    with cost_timer(cores=1, mem_gib=2) as t:
        time.sleep(0.005)
    assert t.cost > 0.0  # CPU + memoria siguen costando aunque no haya GPU


def test_cost_counted_even_on_exception():
    timer = CostTimer(gpu="L4")
    with pytest.raises(RuntimeError):
        with timer:
            time.sleep(0.005)
            raise RuntimeError("boom")
    # El trabajo fallido también se factura: el tiempo quedó registrado.
    assert timer.seconds >= 0.005
    assert timer.cost > 0.0


def test_as_dict_shape():
    with cost_timer(gpu="T4") as t:
        time.sleep(0.001)
    d = t.as_dict()
    assert set(d) == {"compute_s", "rate_usd_per_s", "cost_usd"}
    assert d["compute_s"] >= 0.0
