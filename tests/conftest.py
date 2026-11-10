"""Shared fixtures. Everything here is SYNTHETIC generated data."""

from __future__ import annotations

import pytest

from drone_rf.manifest import build_manifest_from_dir
from drone_rf.splits import assign_splits
from drone_rf.synthetic import make_fixture, make_fixture_tar


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    return make_fixture(tmp_path_factory.mktemp("fx") / "data", samples_per_tag=4, size=48)


@pytest.fixture(scope="session")
def edge_dir(tmp_path_factory):
    return make_fixture(
        tmp_path_factory.mktemp("edge") / "data", samples_per_tag=2, size=32, with_edge_cases=True
    )


@pytest.fixture(scope="session")
def fixture_tar(fixture_dir, tmp_path_factory):
    return make_fixture_tar(fixture_dir, tmp_path_factory.mktemp("tar") / "fixture.tar")


@pytest.fixture(scope="session")
def manifest(fixture_dir):
    return build_manifest_from_dir(fixture_dir)[0]


@pytest.fixture(scope="session")
def splits(manifest):
    return assign_splits(manifest, seed=42)
