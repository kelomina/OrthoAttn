from src.dsra.swanlab_utils import init_swanlab, SwanLabRunProxy


def test_proxy_disabled_is_noop():
    proxy = SwanLabRunProxy(enabled=False)
    proxy.log({"loss": 0.5}, step=1)
    proxy.finish()
    assert proxy.id is None


def test_init_swanlab_disabled_mode():
    run = init_swanlab(mode="disabled")
    assert not run.enabled


def test_swanlab_import_succeeds():
    import swanlab
    assert swanlab is not None
