"""Tests for shared constants."""


def test_bscan_shape_is_tuple_of_two_ints():
    from simtodata.data.constants import BSCAN_SHAPE
    assert isinstance(BSCAN_SHAPE, tuple)
    assert len(BSCAN_SHAPE) == 2
    assert all(isinstance(x, int) for x in BSCAN_SHAPE)


def test_bscan_shape_default():
    from simtodata.data.constants import BSCAN_SHAPE
    assert BSCAN_SHAPE == (64, 64)
