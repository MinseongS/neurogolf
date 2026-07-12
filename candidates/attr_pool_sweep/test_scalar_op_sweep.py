import numpy as np

from candidates.attr_pool_sweep.scalar_op_sweep import (
    OpConfig,
    apply_config,
    build_model,
)


def test_cumsum_directions_and_exclusive_mode():
    source = np.zeros((1, 5, 5), dtype=bool)
    source[0, 2, 1] = True
    forward = apply_config(source, OpConfig("cumsum", axis=2))
    assert forward[0, 2, 1:].all()
    reverse = apply_config(source, OpConfig("cumsum", axis=2, reverse=1))
    assert reverse[0, 2, :2].all()
    exclusive = apply_config(source, OpConfig("cumsum", axis=2, exclusive=1))
    assert not exclusive[0, 2, 1] and exclusive[0, 2, 2:].all()


def test_reverse_sequence_and_trilu():
    source = np.zeros((1, 5, 5), dtype=bool)
    source[0, 1, 3] = True
    reversed_rows = apply_config(source, OpConfig("reverse_sequence", axis=1))
    assert reversed_rows[0, 3, 3]

    full = np.ones((1, 5, 5), dtype=bool)
    upper = apply_config(full, OpConfig("trilu", upper=1, k=0))
    assert np.array_equal(upper[0], np.triu(np.ones((5, 5), dtype=bool)))


def test_scalar_op_onnx_models_have_one_counted_parameter():
    for config in [
        OpConfig("cumsum", axis=3, reverse=1),
        OpConfig("reverse_sequence", axis=2),
        OpConfig("trilu", upper=0, k=-2),
    ]:
        model = build_model(config)
        assert len(model.graph.node) == 1
        assert sum(max(1, int(np.prod(init.dims))) for init in model.graph.initializer) == 1
