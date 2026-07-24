from tim_2026.ablations import PECT_ABLATIONS, apply_ablation
from tim_2026.config import ModelConfig


def test_pect_suite_has_exactly_fourteen_unique_variants() -> None:
    names = [spec.name for spec in PECT_ABLATIONS]
    assert len(names) == 14
    assert len(set(names)) == 14


def test_ablation_changes_only_declared_fields() -> None:
    base = ModelConfig()
    for spec in PECT_ABLATIONS:
        changed = apply_ablation(base, spec.name)
        actual = {
            key
            for key in spec.overrides
            if getattr(changed, key) != getattr(base, key)
        }
        expected = {
            key
            for key, value in spec.overrides.items()
            if value != getattr(base, key)
        }
        assert actual == expected
