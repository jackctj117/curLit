"""Independent inventory vectors for the strict CI advisory manifest."""

import pytest
from scripts.ci_audit_manifest import audit_requirements


def test_all_third_party_packages_retained_with_explicit_cpu_mapping() -> None:
    result = audit_requirements([
        ("curlit", "0.1.0"), ("Torch", "2.11.0+cpu"),
        ("pip", "26.0.1"), ("Foo_Bar", "1.2.3rc1"),
    ])
    # Preserve even a known vulnerable pip version for the auditor to reject.
    assert result.splitlines()[2:] == ["foo-bar==1.2.3rc1", "pip==26.0.1", "torch==2.11.0"]
    assert "torch==2.11.0+cpu -> 2.11.0" in result


@pytest.mark.parametrize("inventory", [
    [("torch", "2.11.0+unreviewed")], [("other", "1.0+cpu")],
    [("foo", "1"), ("Foo", "2")], [("bad\nname", "1")],
    [("foo", "not-a-version")], [("curlit", "0.1.0")], [],
])
def test_ambiguous_or_empty_inventory_blocks_audit(inventory: list[tuple[str, str]]) -> None:
    with pytest.raises(ValueError):
        audit_requirements(inventory)
