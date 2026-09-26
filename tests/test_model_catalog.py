import pytest

from haa.model_catalog import MODEL_CATALOG, definition_for_label, implementations, resolve, strategies, variants
from haa.strategies import HAASimpleIsrael


def test_catalog_exposes_every_stable_model_once():
    labels = [item.label for item in MODEL_CATALOG]
    assert len(labels) == len(set(labels))
    assert definition_for_label("HAA-Simple Israel").model_class is HAASimpleIsrael


def test_israel_implementation_is_only_available_for_simple_haa():
    assert implementations("HAA", "Simple") == ("Original", "Israel")
    assert implementations("HAA", "HAA 4") == ("Original",)
    assert implementations("Inflation Compass", "Steady (80-day)") == ("Original",)


def test_catalog_selection_resolves_existing_stable_model_class():
    assert strategies() == ("HAA", "Inflation Compass")
    assert "HAA 4" in variants("HAA")
    assert resolve("HAA", "Simple", "Israel").model_class is HAASimpleIsrael
    with pytest.raises(ValueError, match="Unknown model selection"):
        resolve("HAA", "HAA 4", "Israel")
