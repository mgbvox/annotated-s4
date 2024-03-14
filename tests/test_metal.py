import jax


def test_device():
    out = jax.devices()
    assert out[0].device_kind == "Metal"
