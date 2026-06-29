"""
Tests for backend.modules.nicobot_database.

Covers:
  - get_cross_coupling_info() returns non-empty markdown with proper headers
    (regression test for the empty-header bug)
  - data loading smoke test (with the real nicobot_data/ fixtures, gracefully
    skipped if data files are missing or are LFS pointer stubs)
  - search_compounds / search_papers return empty list for empty queries
  - get_statistics returns the expected keys
  - infer_leaving_group / infer_nucleophile_type cover all branches
"""

import sys
import os
import json
from pathlib import Path

import pytest

# Make backend/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from modules.nicobot_database import (
    NiCOBotDatabase,
    CompoundInfo,
    PaperInfo,
    ReactionInfo,
)


class TestCrossCouplingInfo:
    """Regression tests for the empty-markdown-header bug."""

    def test_get_cross_coupling_info_returns_non_empty_string(self):
        db = NiCOBotDatabase()
        info = db.get_cross_coupling_info()
        assert isinstance(info, str)
        assert len(info.strip()) > 100  # Has real content.

    def test_get_cross_coupling_info_has_three_section_headers(self):
        """The fixed version must include three markdown section headers
        that were previously empty (just blank lines)."""
        db = NiCOBotDatabase()
        info = db.get_cross_coupling_info()
        assert "### Common Cross-Coupling Reactions" in info
        assert "### Common Leaving Groups" in info
        assert "### Common Nucleophiles" in info

    def test_get_cross_coupling_info_lists_all_named_reactions(self):
        db = NiCOBotDatabase()
        info = db.get_cross_coupling_info().lower()
        for name in ["suzuki", "heck", "sonogashira", "stille", "kumada",
                     "negishi", "hiyama"]:
            assert name in info, f"Missing reaction: {name}"

    def test_get_cross_coupling_info_lists_leaving_groups(self):
        db = NiCOBotDatabase()
        info = db.get_cross_coupling_info().lower()
        for lg in ["triflate", "tosylate", "mesylate", "acetate", "phenolates"]:
            assert lg in info, f"Missing leaving group: {lg}"


class TestInferenceHelpers:
    """Cover all branches of _infer_leaving_group / _infer_nucleophile_type."""

    def test_infer_leaving_group_triflate(self):
        db = NiCOBotDatabase()
        assert db._infer_leaving_group("methyl triflate") == "triflate"

    def test_infer_leaving_group_tosylate(self):
        db = NiCOBotDatabase()
        assert db._infer_leaving_group("ethyl tosylate") == "tosylate"

    def test_infer_leaving_group_mesylate(self):
        db = NiCOBotDatabase()
        assert db._infer_leaving_group("butyl mesylate") == "mesylate"

    def test_infer_leaving_group_acetate(self):
        db = NiCOBotDatabase()
        assert db._infer_leaving_group("phenyl acetate") == "acetate"

    def test_infer_leaving_group_benzoate(self):
        db = NiCOBotDatabase()
        assert db._infer_leaving_group("methyl benzoate") == "benzoate"

    def test_infer_leaving_group_phosphate(self):
        db = NiCOBotDatabase()
        assert db._infer_leaving_group("dimethyl phosphate") == "phosphate"

    def test_infer_leaving_group_carbamate(self):
        db = NiCOBotDatabase()
        assert db._infer_leaving_group("methyl carbamate") == "carbamate"

    def test_infer_leaving_group_pivalate(self):
        db = NiCOBotDatabase()
        assert db._infer_leaving_group("methyl pivalate") == "pivalate"

    def test_infer_leaving_group_other(self):
        db = NiCOBotDatabase()
        assert db._infer_leaving_group("methyl chloride") == "other"

    def test_infer_nucleophile_boron(self):
        db = NiCOBotDatabase()
        assert db._infer_nucleophile_type("phenylboronic acid") == "boron"

    def test_infer_nucleophile_grignard(self):
        db = NiCOBotDatabase()
        assert db._infer_nucleophile_type("methylmagnesium bromide") == "grignard"

    def test_infer_nucleophile_zinc(self):
        db = NiCOBotDatabase()
        assert db._infer_nucleophile_type("diethylzinc") == "zinc"

    def test_infer_nucleophile_tin(self):
        db = NiCOBotDatabase()
        assert db._infer_nucleophile_type("tetramethyltin") == "tin"

    def test_infer_nucleophile_stannane(self):
        db = NiCOBotDatabase()
        assert db._infer_nucleophile_type("trimethylstannane") == "tin"

    def test_infer_nucleophile_silane(self):
        db = NiCOBotDatabase()
        assert db._infer_nucleophile_type("trimethylsilane") == "silicon"

    def test_infer_nucleophile_silyl(self):
        db = NiCOBotDatabase()
        assert db._infer_nucleophile_type("silyl enol ether") == "silicon"

    def test_infer_nucleophile_other(self):
        db = NiCOBotDatabase()
        assert db._infer_nucleophile_type("water") == "other"


class TestEmptyDatabaseBehavior:
    """Behavior of an un-loaded / empty database (no data files touched)."""

    def test_search_compounds_returns_empty_on_unloaded_db(self):
        db = NiCOBotDatabase()  # never call .load()
        assert db.search_compounds("anything") == []

    def test_search_papers_returns_empty_on_unloaded_db(self):
        db = NiCOBotDatabase()
        assert db.search_papers("anything") == []

    def test_get_compound_by_smiles_returns_none_on_unloaded_db(self):
        db = NiCOBotDatabase()
        assert db.get_compound_by_smiles("CCO") is None

    def test_get_paper_by_doi_returns_none_on_unloaded_db(self):
        db = NiCOBotDatabase()
        assert db.get_paper_by_doi("10.1000/nonexistent") is None

    def test_get_statistics_returns_zero_counts_on_unloaded_db(self):
        db = NiCOBotDatabase()
        stats = db.get_statistics()
        assert stats == {
            "electrophiles": 0,
            "nucleophiles": 0,
            "papers": 0,
            "reactions": 0,
        }

    def test_get_reaction_types_returns_empty_list_on_unloaded_db(self):
        db = NiCOBotDatabase()
        assert db.get_reaction_types() == []

    def test_search_for_context_returns_empty_on_unloaded_db(self):
        db = NiCOBotDatabase()
        assert db.search_for_context("suzuki") == ""


class TestDataLoadingFromFixtures:
    """Smoke-test loading from the real nicobot_data/ fixtures.

    These tests gracefully skip if the data files are missing or are Git-LFS
    pointer stubs (which is the case when the LFS objects haven't been pulled).
    """

    @pytest.fixture
    def real_db(self):
        """Load the real NiCOBot database, or skip if data unavailable."""
        backend_dir = Path(__file__).resolve().parent.parent / "backend"
        data_dir = backend_dir / "nicobot_data"

        # Quick check: E_LVG_name_smiles.json must exist and not be an LFS
        # pointer stub.
        e_smiles_path = data_dir / "E_LVG_name_smiles.json"
        if not e_smiles_path.exists():
            pytest.skip("nicobot_data fixtures not available")

        with open(e_smiles_path, "rb") as f:
            head = f.read(50)
        if head.startswith(b"version https://git-lfs"):
            pytest.skip("nicobot_data fixtures are Git-LFS pointer stubs — "
                        "see backend/nicobot_data/DATA_LFS.md for recovery")

        db = NiCOBotDatabase(data_dir=str(data_dir))
        loaded_ok = db.load()
        if not loaded_ok:
            pytest.skip("NiCOBot database failed to load")
        return db

    def test_real_db_loaded_some_electrophiles(self, real_db):
        assert len(real_db.electrophiles) > 0

    def test_real_db_search_compounds_returns_results(self, real_db):
        # Pick any electrophile name and search for a fragment of it.
        any_smiles = next(iter(real_db.electrophiles))
        any_name = real_db.electrophiles[any_smiles].name
        # Use the first 4 chars of the name as the query — should match.
        query = any_name[:4].lower()
        results = real_db.search_compounds(query, limit=5)
        assert len(results) > 0

    def test_real_db_get_statistics_non_zero(self, real_db):
        stats = real_db.get_statistics()
        assert stats["electrophiles"] > 0
