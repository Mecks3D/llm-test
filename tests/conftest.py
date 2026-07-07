"""Salta i test marcati @pytest.mark.torch se PyTorch non è installato
(FASE2_PIANO.md §9): niente errori di import in ambienti senza torch."""
import pytest

try:
    import torch  # noqa: F401
    _TORCH_DISPONIBILE = True
except ImportError:
    _TORCH_DISPONIBILE = False


def pytest_collection_modifyitems(config, items):
    if _TORCH_DISPONIBILE:
        return
    skip_torch = pytest.mark.skip(reason="torch non installato")
    for item in items:
        if "torch" in item.keywords:
            item.add_marker(skip_torch)
