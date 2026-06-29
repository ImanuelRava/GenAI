"""
Tests for the decomposed chemextract package.

Verifies:
  - The split modules (vision_pipeline, text_pipeline, post_process) import
    cleanly and expose the expected classes / functions.
  - ChemExtractAI inherits from all 3 mixins and retains its full public API.
  - The module-level pure functions in post_process.py work correctly:
    - clean_reaction_smiles (pseudo-SMILES → general_form)
    - clean_compound_smiles
    - deduplicate_compounds (by name-or-SMILES)
    - deduplicate_reactions (by Jaccard overlap)
    - normalize_reactions (key defaults, outcomes.yield hoist, conditions sort)
    - normalize_compounds (key defaults, SMILES strip)
  - The result-merging mixin methods (_merge_figure_result, _merge_vision_results)
    correctly translate the structured + flat LLM output formats into the
    unified result dict shape.
  - The _guess_figure_type heuristic returns sensible hints for various
    width/height combinations.
  - _make_empty_result produces the correct initial shape (used by both
    sync + async entry points, so cache consistency depends on it).
  - The top-level convenience wrappers (extract_chemical_data_from_pdf,
    extract_chemical_data_from_pdf_async) are still callable + delegate to
    ChemExtractAI correctly (verified with mocked vision/text pipelines).
"""

import sys
import os
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any

import pytest

# Make backend/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from modules.chemextract.extractor import (
    ChemExtractAI,
    extract_chemical_data_from_pdf,
    extract_chemical_data_from_pdf_async,
)
from modules.chemextract.vision_pipeline import VisionPipelineMixin
from modules.chemextract.text_pipeline import TextPipelineMixin
from modules.chemextract.post_process import (
    PostProcessMixin,
    clean_reaction_smiles,
    clean_compound_smiles,
    deduplicate_compounds,
    deduplicate_reactions,
    normalize_reactions,
    normalize_compounds,
)


# =============================================================================
# Module structure tests
# =============================================================================

class TestModuleStructure:
    """Verify the package layout is correct and all modules import cleanly."""

    def test_vision_pipeline_module_importable(self):
        from modules.chemextract import vision_pipeline
        assert hasattr(vision_pipeline, 'VisionPipelineMixin')

    def test_text_pipeline_module_importable(self):
        from modules.chemextract import text_pipeline
        assert hasattr(text_pipeline, 'TextPipelineMixin')

    def test_post_process_module_importable(self):
        from modules.chemextract import post_process
        assert hasattr(post_process, 'PostProcessMixin')
        # Module-level pure functions
        for fn_name in ['clean_reaction_smiles', 'clean_compound_smiles',
                        'deduplicate_compounds', 'deduplicate_reactions',
                        'normalize_reactions', 'normalize_compounds']:
            assert hasattr(post_process, fn_name), f"Missing {fn_name}"

    def test_chemextract_ai_inherits_all_three_mixins(self):
        """ChemExtractAI must inherit from VisionPipelineMixin, TextPipelineMixin,
        and PostProcessMixin so all the pipeline methods are available."""
        assert issubclass(ChemExtractAI, VisionPipelineMixin)
        assert issubclass(ChemExtractAI, TextPipelineMixin)
        assert issubclass(ChemExtractAI, PostProcessMixin)

    def test_chemextract_init_re_exports_unchanged(self):
        """chemextract/__init__.py must still export the same symbols."""
        from modules.chemextract import (
            ChemExtractAI as CE,
            extract_chemical_data_from_pdf as sync_fn,
            extract_chemical_data_from_pdf_async as async_fn,
        )
        assert CE is ChemExtractAI
        assert sync_fn is extract_chemical_data_from_pdf
        assert async_fn is extract_chemical_data_from_pdf_async


class TestChemExtractAIPublicAPI:
    """Verify the ChemExtractAI class retains its full original public API."""

    @pytest.fixture
    def instance(self):
        return ChemExtractAI(llm_provider='deepseek', api_key='sk-test')

    def test_constructor_sets_attributes(self, instance):
        assert instance.llm_provider == 'deepseek'
        assert instance.api_key == 'sk-test'
        assert instance.model == 'deepseek-chat'  # default for deepseek
        assert instance.vision_providers == ['deepseek', 'openai', 'gemini', 'anthropic']

    def test_constructor_resolves_default_model_per_provider(self):
        for provider, expected_model in [
            ('deepseek', 'deepseek-chat'),
            ('openai', 'gpt-4o'),
            ('gemini', 'gemini-2.0-flash'),
            ('anthropic', 'claude-3-5-sonnet-20241022'),
        ]:
            inst = ChemExtractAI(llm_provider=provider, api_key='sk-test')
            assert inst.model == expected_model, f"{provider}: {inst.model}"

    def test_explicit_model_overrides_default(self):
        inst = ChemExtractAI(llm_provider='deepseek', api_key='sk-test', model='custom-model')
        assert inst.model == 'custom-model'

    def test_all_original_methods_present(self, instance):
        """All 24 methods from the original monolithic class must still be present."""
        original_methods = [
            'extract_from_pdf', 'extract_from_pdf_async',
            '_run_vision_pipeline', '_run_vision_pipeline_async',
            '_fallback_full_page_vision',
            '_analyze_embedded_figure', '_analyze_embedded_figure_async',
            '_analyze_scheme_page', '_analyze_scheme_page_async',
            '_guess_figure_type',
            '_extract_from_image', '_extract_from_image_async',
            '_run_text_pipeline', '_run_text_pipeline_async',
            '_merge_figure_result', '_merge_vision_results',
            '_post_process',
            '_clean_reaction_smiles', '_clean_compound_smiles',
            '_deduplicate_compounds', '_deduplicate_reactions',
            '_normalize_reactions', '_normalize_compounds',
            '_make_empty_result',
        ]
        missing = [m for m in original_methods if not hasattr(instance, m)]
        assert missing == [], f"Missing methods: {missing}"


# =============================================================================
# _make_empty_result tests
# =============================================================================

class TestMakeEmptyResult:
    """Verify the empty-result helper produces the correct shape."""

    def test_returns_dict_with_all_top_level_keys(self):
        inst = ChemExtractAI(llm_provider='deepseek', api_key='sk-test')
        result = inst._make_empty_result()
        for key in ['reactions', 'compounds', 'figures', 'tables', 'text_content', 'metadata']:
            assert key in result, f"Missing key: {key}"

    def test_top_level_collections_start_empty(self):
        inst = ChemExtractAI(llm_provider='deepseek', api_key='sk-test')
        result = inst._make_empty_result()
        assert result['reactions'] == []
        assert result['compounds'] == []
        assert result['figures'] == []
        assert result['tables'] == []
        assert result['text_content'] == ''

    def test_metadata_has_expected_fields(self):
        inst = ChemExtractAI(llm_provider='deepseek', api_key='sk-test')
        result = inst._make_empty_result()
        meta = result['metadata']
        assert meta['extraction_method'] == 'chemextract_ai'
        assert meta['provider'] == 'deepseek'
        assert meta['model'] == 'deepseek-chat'
        assert meta['text_extracted'] is False
        assert meta['images_extracted'] is False
        assert meta['embedded_figures_found'] == 0
        assert meta['scheme_pages_found'] == 0
        assert meta['vision_mode'] == 'figure_extraction'

    def test_metadata_reflects_provider_and_model(self):
        inst = ChemExtractAI(llm_provider='openai', api_key='sk-test', model='gpt-4o')
        result = inst._make_empty_result()
        assert result['metadata']['provider'] == 'openai'
        assert result['metadata']['model'] == 'gpt-4o'


# =============================================================================
# clean_reaction_smiles tests (module-level pure function)
# =============================================================================

class TestCleanReactionSmiles:
    """Test the pseudo-SMILES cleanup for reactions."""

    def test_empty_list_returns_empty(self):
        assert clean_reaction_smiles([]) == []

    def test_passes_through_non_dict_entries(self):
        result = clean_reaction_smiles(["not a dict", 42, None])
        assert result == ["not a dict", 42, None]

    def test_real_smiles_preserved(self):
        reactions = [{
            "reactants": [{"name": "bromobenzene", "smiles": "c1ccc(Br)cc1"}],
            "products": [{"name": "biphenyl", "smiles": "c1ccc(-c2ccccc2)cc1"}],
        }]
        result = clean_reaction_smiles(reactions)
        assert result[0]["reactants"][0]["smiles"] == "c1ccc(Br)cc1"
        assert result[0]["products"][0]["smiles"] == "c1ccc(-c2ccccc2)cc1"
        # No general_form added for real SMILES
        assert "general_form" not in result[0]["reactants"][0]

    def test_pseudo_smiles_moved_to_general_form(self):
        """Pseudo-SMILES like 'R1-I' should be moved to general_form, smiles=None."""
        reactions = [{
            "reactants": [{"name": "R1-iodide", "smiles": "R1-I"}],
            "products": [{"name": "product", "smiles": "c1ccccc1"}],
        }]
        result = clean_reaction_smiles(reactions)
        assert result[0]["reactants"][0]["smiles"] is None
        assert result[0]["reactants"][0]["general_form"] == "R1-I"
        # Real SMILES in products unchanged
        assert result[0]["products"][0]["smiles"] == "c1ccccc1"

    def test_pseudo_smiles_in_catalysts_nulled_without_general_form(self):
        """Catalysts/ligands pseudo-SMILES are nulled but don't get general_form.

        The _is_pseudo_smiles detector matches 'R' or 'Ar' at a word boundary
        followed by [A-Z0-9[(]. We use 'R1Pd' (R at start, followed by digit)
        and 'ArL' (Ar at start, followed by uppercase letter) to trigger it.
        """
        reactions = [{
            "catalysts": [{"name": "Pd", "smiles": "R1Pd"}],
            "ligands": [{"name": "L", "smiles": "ArL"}],
        }]
        result = clean_reaction_smiles(reactions)
        assert result[0]["catalysts"][0]["smiles"] is None
        assert "general_form" not in result[0]["catalysts"][0]
        assert result[0]["ligands"][0]["smiles"] is None
        assert "general_form" not in result[0]["ligands"][0]

    def test_reaction_dict_shallow_copied(self):
        """The function returns new reaction dicts (shallow copy), but the
        entity dicts inside ARE mutated in place — this matches the original
        code's behavior. Callers must deep-copy if they need to preserve
        the original entity dicts."""
        original = [{
            "reactants": [{"name": "x", "smiles": "R1-I"}],
            "type": "Suzuki",
        }]
        result = clean_reaction_smiles(original)
        # The reaction dict itself is a new object (shallow copy).
        assert result[0] is not original[0]
        # Original reaction's 'type' is preserved (not mutated).
        assert original[0]["type"] == "Suzuki"
        # The entity dict inside IS mutated (matches original behavior).
        assert original[0]["reactants"][0]["smiles"] is None
        assert original[0]["reactants"][0]["general_form"] == "R1-I"


# =============================================================================
# clean_compound_smiles tests
# =============================================================================

class TestCleanCompoundSmiles:

    def test_empty_list_returns_empty(self):
        assert clean_compound_smiles([]) == []

    def test_real_smiles_preserved(self):
        compounds = [{"name": "benzene", "smiles": "c1ccccc1"}]
        result = clean_compound_smiles(compounds)
        assert result[0]["smiles"] == "c1ccccc1"
        assert "general_form" not in result[0]

    def test_pseudo_smiles_moved_to_general_form(self):
        compounds = [{"name": "R-group", "smiles": "R1-Br"}]
        result = clean_compound_smiles(compounds)
        assert result[0]["smiles"] is None
        assert result[0]["general_form"] == "R1-Br"


# =============================================================================
# deduplicate_compounds tests
# =============================================================================

class TestDeduplicateCompounds:

    def test_empty_list_returns_empty(self):
        assert deduplicate_compounds([]) == []

    def test_dedup_by_name_case_insensitive(self):
        compounds = [
            {"name": "Benzene", "smiles": "c1ccccc1"},
            {"name": "benzene", "smiles": "c1ccccc1"},  # dup (case-insensitive)
            {"name": "Toluene", "smiles": "Cc1ccccc1"},
        ]
        result = deduplicate_compounds(compounds)
        assert len(result) == 2
        assert result[0]["name"] == "Benzene"  # first occurrence wins
        assert result[1]["name"] == "Toluene"

    def test_dedup_by_smiles_when_name_missing(self):
        """When name is empty, dedup falls back to lowercased SMILES.
        Note: 'c1ccccc1' and 'C1CCCCC1' both lowercase to 'c1ccccc1', so
        they're treated as duplicates — this matches the original behavior."""
        compounds = [
            {"name": "", "smiles": "c1ccccc1"},
            {"name": "", "smiles": "Cc1ccccc1"},  # different SMILES (toluene), kept
            {"name": "", "smiles": "c1ccccc1"},   # dup of first
        ]
        result = deduplicate_compounds(compounds)
        assert len(result) == 2
        assert result[0]["smiles"] == "c1ccccc1"
        assert result[1]["smiles"] == "Cc1ccccc1"

    def test_compounds_with_no_name_or_smiles_dropped(self):
        """Compounds with neither name nor SMILES are DROPPED entirely.

        The dedup function's `if key` check skips compounds whose key is
        empty (both name and smiles missing). This matches the original
        behavior — such compounds provide no useful identifying info."""
        compounds = [
            {"name": "", "smiles": ""},
            {"name": "", "smiles": ""},
            {"name": "benzene", "smiles": "c1ccccc1"},
        ]
        result = deduplicate_compounds(compounds)
        assert len(result) == 1
        assert result[0]["name"] == "benzene"


# =============================================================================
# deduplicate_reactions tests
# =============================================================================

class TestDeduplicateReactions:

    def test_empty_list_returns_empty(self):
        assert deduplicate_reactions([]) == []

    def test_keeps_non_overlapping_reactions(self):
        reactions = [
            {"reactants": [{"name": "A"}], "products": [{"name": "B"}]},
            {"reactants": [{"name": "C"}], "products": [{"name": "D"}]},
        ]
        result = deduplicate_reactions(reactions)
        assert len(result) == 2

    def test_drops_near_duplicate_reactions(self):
        """Reactions with >80% Jaccard overlap in both reactants + products
        are considered duplicates."""
        reactions = [
            {"reactants": [{"name": "A"}, {"name": "B"}], "products": [{"name": "C"}]},
            {"reactants": [{"name": "A"}, {"name": "B"}], "products": [{"name": "C"}]},
        ]
        result = deduplicate_reactions(reactions)
        assert len(result) == 1

    def test_keeps_reactions_with_no_named_entities(self):
        """Reactions with no named reactants/products are kept as-is."""
        reactions = [
            {"reactants": [], "products": []},
            {"reactants": [], "products": []},
        ]
        result = deduplicate_reactions(reactions)
        assert len(result) == 2

    def test_dedup_is_case_insensitive(self):
        reactions = [
            {"reactants": [{"name": "Bromobenzene"}], "products": [{"name": "Biphenyl"}]},
            {"reactants": [{"name": "bromobenzene"}], "products": [{"name": "biphenyl"}]},
        ]
        result = deduplicate_reactions(reactions)
        assert len(result) == 1


# =============================================================================
# normalize_reactions tests
# =============================================================================

class TestNormalizeReactions:

    def test_empty_list_returns_empty(self):
        assert normalize_reactions([]) == []

    def test_sets_default_keys(self):
        reactions = [{"id": "r1"}]
        result = normalize_reactions(reactions)
        assert result[0]["id"] == "r1"
        assert result[0]["type"] == "unknown"
        assert result[0]["conditions"] == {}
        assert result[0]["reactants"] == []
        assert result[0]["products"] == []

    def test_hoists_outcomes_yield_to_top_level(self):
        reactions = [{"outcomes": {"yield": "85%"}}]
        result = normalize_reactions(reactions)
        assert result[0]["yield"] == "85%"

    def test_does_not_overwrite_explicit_yield(self):
        reactions = [{"yield": "80%", "outcomes": {"yield": "85%"}}]
        result = normalize_reactions(reactions)
        assert result[0]["yield"] == "80%"

    def test_derives_entry_id_from_entry(self):
        reactions = [{"entry": 5}]
        result = normalize_reactions(reactions)
        assert result[0]["entry_id"] == "5"

    def test_sorts_conditions_dict(self):
        reactions = [{"conditions": {"temperature": "80 C", "time": "12 h", "atmosphere": "N2"}}]
        result = normalize_reactions(reactions)
        # Sorted keys: atmosphere, temperature, time
        assert list(result[0]["conditions"].keys()) == ["atmosphere", "temperature", "time"]


# =============================================================================
# normalize_compounds tests
# =============================================================================

class TestNormalizeCompounds:

    def test_empty_list_returns_empty(self):
        assert normalize_compounds([]) == []

    def test_sets_default_keys(self):
        compounds = [{"name": "benzene"}]
        result = normalize_compounds(compounds)
        assert result[0]["name"] == "benzene"
        assert result[0]["smiles"] is None
        assert result[0]["role"] == "unknown"
        assert result[0]["formula"] is None

    def test_strips_smiles_whitespace(self):
        compounds = [{"name": "benzene", "smiles": "  c1ccccc1  "}]
        result = normalize_compounds(compounds)
        assert result[0]["smiles"] == "c1ccccc1"


# =============================================================================
# _guess_figure_type heuristic tests
# =============================================================================

class TestGuessFigureType:

    def test_wide_figure(self):
        """Aspect ratio > 3.0 → wide/narrow figure."""
        hint = ChemExtractAI._guess_figure_type({"width": 1200, "height": 300})
        assert "wide/narrow" in hint

    def test_tall_figure(self):
        """Aspect ratio < 0.33 → wide/narrow figure."""
        hint = ChemExtractAI._guess_figure_type({"width": 300, "height": 1200})
        assert "wide/narrow" in hint

    def test_large_figure(self):
        """Width or height > 1500 → large figure."""
        hint = ChemExtractAI._guess_figure_type({"width": 2000, "height": 1500})
        assert "large" in hint

    def test_small_icon(self):
        """Width and height both < 300 → small image icon."""
        hint = ChemExtractAI._guess_figure_type({"width": 200, "height": 200})
        assert "small" in hint

    def test_square_figure(self):
        """Aspect ratio between 0.8 and 1.2 → square-ish figure."""
        hint = ChemExtractAI._guess_figure_type({"width": 800, "height": 800})
        assert "square" in hint

    def test_rectangular_figure(self):
        """Default case → rectangular figure."""
        hint = ChemExtractAI._guess_figure_type({"width": 800, "height": 600})
        assert "rectangular" in hint

    def test_zero_height_safe(self):
        """Height=0 should not cause a ZeroDivisionError (max(h, 1) guard)."""
        hint = ChemExtractAI._guess_figure_type({"width": 800, "height": 0})
        assert isinstance(hint, str)
        assert len(hint) > 0


# =============================================================================
# Result-merging mixin tests
# =============================================================================

class TestMergeFigureResult:
    """Test the _merge_figure_result method (structured format from figure analysis)."""

    @pytest.fixture
    def instance(self):
        return ChemExtractAI(llm_provider='deepseek', api_key='sk-test')

    @pytest.fixture
    def empty_result(self, instance):
        return instance._make_empty_result()

    def test_merges_structured_reaction_schemes(self, instance, empty_result):
        """The SYSTEM_PROMPT_FIGURE_ANALYSIS format has reaction_schemes."""
        vision_data = {
            "reaction_schemes": [{
                "reactionType": "Suzuki",
                "reactants": [{"name": "bromobenzene", "smiles": "c1ccc(Br)cc1"}],
                "products": [{"name": "biphenyl", "smiles": "c1ccc(-c2ccccc2)cc1"}],
                "reagents": ["Pd(PPh3)4"],
                "conditions": {"temperature": "80 C"},
                "yield": "85%",
            }],
        }
        instance._merge_figure_result(empty_result, vision_data, page_num=1, source="embedded")

        assert len(empty_result["reactions"]) == 1
        rxn = empty_result["reactions"][0]
        assert rxn["type"] == "Suzuki"
        assert rxn["page"] == 1
        assert rxn["source"] == "embedded"
        assert rxn["yield"] == "85%"
        assert rxn["id"].startswith("embedded_page1_")

    def test_extracts_compounds_from_schemes(self, instance, empty_result):
        """Compounds are extracted from reaction scheme reactants + products."""
        vision_data = {
            "reaction_schemes": [{
                "reactants": [{"name": "bromobenzene", "smiles": "c1ccc(Br)cc1"}],
                "products": [{"name": "biphenyl", "smiles": "c1ccc(-c2ccccc2)cc1"}],
            }],
        }
        instance._merge_figure_result(empty_result, vision_data, page_num=1, source="embedded")

        names = {c["name"] for c in empty_result["compounds"]}
        assert "bromobenzene" in names
        assert "biphenyl" in names
        # Roles assigned correctly
        for comp in empty_result["compounds"]:
            if comp["name"] == "bromobenzene":
                assert comp["role"] == "reactant"
            elif comp["name"] == "biphenyl":
                assert comp["role"] == "product"

    def test_dedup_compounds_within_merge(self, instance, empty_result):
        """If the same compound appears in multiple schemes, it's deduped at merge time."""
        vision_data = {
            "reaction_schemes": [
                {"reactants": [{"name": "bromobenzene"}], "products": []},
                {"reactants": [{"name": "bromobenzene"}], "products": []},
            ],
        }
        instance._merge_figure_result(empty_result, vision_data, page_num=1, source="embedded")
        # Only one bromobenzene entry
        benzene_entries = [c for c in empty_result["compounds"] if c["name"] == "bromobenzene"]
        assert len(benzene_entries) == 1

    def test_falls_back_to_flat_format(self, instance, empty_result):
        """If vision_data has no reaction_schemes but has flat reactants/products,
        _merge_figure_result should delegate to _merge_vision_results."""
        vision_data = {
            "reactants": ["bromobenzene"],
            "products": ["biphenyl"],
        }
        instance._merge_figure_result(empty_result, vision_data, page_num=1, source="embedded")

        # Should still produce a reaction (via the flat-format fallback).
        assert len(empty_result["reactions"]) == 1
        assert len(empty_result["compounds"]) >= 2

    def test_records_figure_metadata(self, instance, empty_result):
        """Figure metadata (type, description) should be recorded in result['figures']."""
        vision_data = {
            "reaction_schemes": [],
            "figure_type": "reaction_scheme",
            "description": "A Suzuki coupling scheme",
            "notes": "Page 1, top right",
        }
        instance._merge_figure_result(empty_result, vision_data, page_num=1, source="embedded")

        assert len(empty_result["figures"]) == 1
        fig = empty_result["figures"][0]
        assert fig["type"] == "reaction_scheme"
        assert fig["description"] == "A Suzuki coupling scheme"
        assert fig["page"] == 1

    def test_pulls_scaffold_and_rgroup_data(self, instance, empty_result):
        """scaffold_smiles / rgroup_table / rgroup_attachment_map from the
        figure analysis are copied to the top-level result."""
        vision_data = {
            "reaction_schemes": [],
            "scaffold_smiles": "c1ccc([*])cc1",
            "rgroup_table": {"R1": {"1a": "Cl"}},
        }
        instance._merge_figure_result(empty_result, vision_data, page_num=1, source="embedded")

        assert empty_result["scaffold_smiles"] == "c1ccc([*])cc1"
        assert empty_result["rgroup_table"] == {"R1": {"1a": "Cl"}}


class TestMergeVisionResults:
    """Test the _merge_vision_results method (legacy flat format from full-page vision)."""

    @pytest.fixture
    def instance(self):
        return ChemExtractAI(llm_provider='deepseek', api_key='sk-test')

    @pytest.fixture
    def empty_result(self, instance):
        return instance._make_empty_result()

    def test_merges_structured_format(self, instance, empty_result):
        """If vision_data has reaction_schemes, use them directly."""
        vision_data = {
            "reaction_schemes": [{
                "reactionType": "Heck",
                "reactants": [{"name": "iodobenzene"}],
                "products": [{"name": "styrene"}],
            }],
        }
        instance._merge_vision_results(empty_result, vision_data, page_num=2)

        assert len(empty_result["reactions"]) == 1
        assert empty_result["reactions"][0]["type"] == "Heck"
        assert empty_result["reactions"][0]["page"] == 2
        assert empty_result["reactions"][0]["source"] == "vision"

    def test_translates_flat_format_to_structured(self, instance, empty_result):
        """If vision_data has flat reactants/products lists, wrap them into
        a single reaction_scheme entry."""
        vision_data = {
            "reactants": ["bromobenzene", "styrene"],
            "products": ["stilbene"],
            "reagents": ["Pd(OAc)2"],
            "conditions": {"temperature": "100 C"},
        }
        instance._merge_vision_results(empty_result, vision_data, page_num=3)

        assert len(empty_result["reactions"]) == 1
        rxn = empty_result["reactions"][0]
        assert len(rxn["reactants"]) == 2
        assert rxn["reactants"][0]["name"] == "bromobenzene"
        assert rxn["reactants"][0]["smiles"] is None  # flat format has no SMILES
        assert rxn["conditions"] == {"temperature": "100 C"}

    def test_records_vision_analysis_figure(self, instance, empty_result):
        """A figure entry with type='vision_analysis' is recorded for each merge."""
        vision_data = {"reaction_schemes": [], "description": "Page analysis"}
        instance._merge_vision_results(empty_result, vision_data, page_num=5)

        assert len(empty_result["figures"]) == 1
        assert empty_result["figures"][0]["type"] == "vision_analysis"
        assert empty_result["figures"][0]["page"] == 5


# =============================================================================
# End-to-end pipeline tests (with mocked vision + text)
# =============================================================================

class TestPipelineOrchestration:
    """Verify the top-level extract_from_pdf correctly orchestrates the mixins.

    These tests mock the vision and text pipeline methods to isolate the
    orchestrator's control flow (cache check → vision → text → post-process →
    cache save) from the actual LLM calls.
    """

    def test_extract_from_pdf_returns_cached_result(self):
        """If a cached result exists, it should be returned without running
        the vision or text pipelines."""
        cached = {"reactions": ["cached"], "compounds": [], "metadata": {}}

        with patch('modules.chemextract.extractor._get_cached_result', return_value=cached):
            instance = ChemExtractAI(llm_provider='deepseek', api_key='sk-test')
            # Mock the pipeline methods to ensure they're NOT called
            instance._run_vision_pipeline = MagicMock()
            instance._run_text_pipeline = MagicMock()
            instance._post_process = MagicMock()

            result = instance.extract_from_pdf('/fake/path.pdf')

            assert result is cached
            instance._run_vision_pipeline.assert_not_called()
            instance._run_text_pipeline.assert_not_called()
            instance._post_process.assert_not_called()

    def test_extract_from_pdf_runs_full_pipeline_when_no_cache(self):
        """Without a cache, all three pipeline steps should run in order."""
        with patch('modules.chemextract.extractor._get_cached_result', return_value=None), \
             patch('modules.chemextract.extractor._save_cached_result') as mock_save:

            instance = ChemExtractAI(llm_provider='deepseek', api_key='sk-test')
            instance._run_vision_pipeline = MagicMock()
            instance._run_text_pipeline = MagicMock()
            instance._post_process = MagicMock()

            result = instance.extract_from_pdf('/fake/path.pdf')

            # All three pipeline steps were called
            instance._run_vision_pipeline.assert_called_once()
            instance._run_text_pipeline.assert_called_once()
            instance._post_process.assert_called_once()
            # Result was saved to cache
            mock_save.assert_called_once()
            # Result is the expected empty-result shape
            assert result["metadata"]["extraction_method"] == "chemextract_ai"

    def test_extract_from_pdf_skips_vision_for_non_vision_provider(self):
        """If the provider doesn't support vision, the vision pipeline is skipped."""
        with patch('modules.chemextract.extractor._get_cached_result', return_value=None), \
             patch('modules.chemextract.extractor._save_cached_result'):

            # groq is not in VISION_PROVIDERS
            instance = ChemExtractAI(llm_provider='groq', api_key='sk-test')
            instance._run_vision_pipeline = MagicMock()
            instance._run_text_pipeline = MagicMock()
            instance._post_process = MagicMock()

            instance.extract_from_pdf('/fake/path.pdf', extract_images=True)

            instance._run_vision_pipeline.assert_not_called()
            instance._run_text_pipeline.assert_called_once()

    def test_extract_from_pdf_respects_extract_images_flag(self):
        """extract_images=False should skip the vision pipeline even for vision providers."""
        with patch('modules.chemextract.extractor._get_cached_result', return_value=None), \
             patch('modules.chemextract.extractor._save_cached_result'):

            instance = ChemExtractAI(llm_provider='deepseek', api_key='sk-test')
            instance._run_vision_pipeline = MagicMock()
            instance._run_text_pipeline = MagicMock()
            instance._post_process = MagicMock()

            instance.extract_from_pdf('/fake/path.pdf', extract_images=False)

            instance._run_vision_pipeline.assert_not_called()
            instance._run_text_pipeline.assert_called_once()

    def test_extract_from_pdf_respects_extract_text_flag(self):
        """extract_text=False should skip the text pipeline."""
        with patch('modules.chemextract.extractor._get_cached_result', return_value=None), \
             patch('modules.chemextract.extractor._save_cached_result'):

            instance = ChemExtractAI(llm_provider='deepseek', api_key='sk-test')
            instance._run_vision_pipeline = MagicMock()
            instance._run_text_pipeline = MagicMock()
            instance._post_process = MagicMock()

            instance.extract_from_pdf('/fake/path.pdf', extract_text=False)

            instance._run_vision_pipeline.assert_called_once()
            instance._run_text_pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_from_pdf_async_returns_cached_result(self):
        cached = {"reactions": ["cached"], "compounds": [], "metadata": {}}
        with patch('modules.chemextract.extractor._get_cached_result', return_value=cached):
            instance = ChemExtractAI(llm_provider='deepseek', api_key='sk-test')
            instance._run_vision_pipeline_async = AsyncMock()
            instance._run_text_pipeline_async = AsyncMock()
            instance._post_process = MagicMock()

            result = await instance.extract_from_pdf_async('/fake/path.pdf')

            assert result is cached
            instance._run_vision_pipeline_async.assert_not_called()
            instance._run_text_pipeline_async.assert_not_called()


# =============================================================================
# Convenience wrapper tests
# =============================================================================

class TestConvenienceWrappers:
    """Verify extract_chemical_data_from_pdf[_async] delegate to ChemExtractAI."""

    def test_sync_wrapper_constructs_instance_and_delegates(self):
        with patch('modules.chemextract.extractor._get_cached_result', return_value=None), \
             patch('modules.chemextract.extractor._save_cached_result'), \
             patch.object(ChemExtractAI, 'extract_from_pdf') as mock_extract:

            mock_extract.return_value = {"reactions": [], "compounds": []}
            result = extract_chemical_data_from_pdf(
                '/fake/path.pdf',
                llm_provider='openai',
                api_key='sk-test',
                model='gpt-4o',
                max_pages=25,
            )

            assert result == {"reactions": [], "compounds": []}
            mock_extract.assert_called_once()
            call_kwargs = mock_extract.call_args
            assert call_kwargs[1]['max_pages'] == 25

    @pytest.mark.asyncio
    async def test_async_wrapper_constructs_instance_and_delegates(self):
        with patch('modules.chemextract.extractor._get_cached_result', return_value=None), \
             patch('modules.chemextract.extractor._save_cached_result'), \
             patch.object(ChemExtractAI, 'extract_from_pdf_async') as mock_extract:

            mock_extract.return_value = {"reactions": [], "compounds": []}
            result = await extract_chemical_data_from_pdf_async(
                '/fake/path.pdf',
                llm_provider='openai',
                api_key='sk-test',
                model='gpt-4o',
            )

            assert result == {"reactions": [], "compounds": []}
            mock_extract.assert_called_once()


# =============================================================================
# Text pipeline mixin tests
# =============================================================================

class TestTextPipelineMixin:
    """Test the TextPipelineMixin._apply_text_data_to_result helper."""

    def test_copies_reactions_and_compounds(self):
        result = {
            "reactions": ["old"],
            "compounds": ["old"],
            "metadata": {},
        }
        text_data = {
            "reactions": [{"id": "r1"}],
            "compounds": [{"name": "benzene"}],
        }
        TextPipelineMixin._apply_text_data_to_result(result, text_data)

        assert result["reactions"] == [{"id": "r1"}]
        assert result["compounds"] == [{"name": "benzene"}]
        assert result["metadata"]["text_extracted"] is True

    def test_copies_optional_keys_when_present(self):
        result = {"reactions": [], "compounds": [], "metadata": {}}
        text_data = {
            "reactions": [],
            "compounds": [],
            "experimental_procedures": ["step 1", "step 2"],
            "characterization_data": {"NMR": "data"},
            "scaffold_smiles": "c1ccc([*])cc1",
            "rgroup_table": {"R1": {"1a": "Cl"}},
        }
        TextPipelineMixin._apply_text_data_to_result(result, text_data)

        assert result["experimental_procedures"] == ["step 1", "step 2"]
        assert result["characterization_data"] == {"NMR": "data"}
        assert result["scaffold_smiles"] == "c1ccc([*])cc1"
        assert result["rgroup_table"] == {"R1": {"1a": "Cl"}}

    def test_does_not_copy_optional_keys_when_absent(self):
        result = {"reactions": [], "compounds": [], "metadata": {}}
        text_data = {"reactions": [], "compounds": []}
        TextPipelineMixin._apply_text_data_to_result(result, text_data)

        assert "experimental_procedures" not in result
        assert "characterization_data" not in result
        assert "scaffold_smiles" not in result
        assert "rgroup_table" not in result
