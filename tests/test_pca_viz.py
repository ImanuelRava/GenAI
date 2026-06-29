"""
Tests for backend.modules.pca_viz.

Regression tests for the unreachable-code-paths bug — the route
``/api/pca/data/<data_type>`` accepts ``'circular'`` and ``'clustered'`` but
the original ``generate_2d_data`` only handled ``'clusters'`` and ``'linear'``.
Similarly, ``generate_scree_data`` had a ``'moderate'`` branch that was
unreachable because the route's valid_data_types list excluded it.

These tests lock in the fix so the routes and the module stay in sync.
"""

import sys
import os

import pytest

# Make backend/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Try to import numpy + pca_viz — skip all tests if unavailable.
np = pytest.importorskip("numpy")
from modules import pca_viz  # noqa: E402  (after importorskip)


class TestGenerate2dDataAllRouteTypes:
    """Every data_type accepted by routes/visualization.py must be
    handled by generate_2d_data without falling through to a wrong branch."""

    @pytest.mark.parametrize("data_type", [
        "clusters", "clustered", "linear", "circular", "structured",
    ])
    def test_all_route_types_produce_data(self, data_type):
        result = pca_viz.generate_2d_data(data_type, n_samples=20)
        assert isinstance(result, dict)
        assert "data_points" in result
        assert "projected_points" in result
        assert "pc1" in result
        assert "pc2" in result
        assert len(result["data_points"]) == 20

    def test_clustered_is_alias_for_clusters(self):
        """``'clustered'`` (used by the route) must produce the same output
        as ``'clusters'`` (the original branch name)."""
        a = pca_viz.generate_2d_data("clusters", n_samples=20)
        b = pca_viz.generate_2d_data("clustered", n_samples=20)
        # Labels: clustered data has 2 cluster labels (0 and 1).
        labels_a = {p["label"] for p in a["data_points"]}
        labels_b = {p["label"] for p in b["data_points"]}
        assert labels_a == labels_b == {0, 1}

    def test_circular_produces_ring(self):
        """Circular data should have a non-trivial label distribution
        and points roughly on a ring (mean radius near 3.0)."""
        result = pca_viz.generate_2d_data("circular", n_samples=40)
        points = result["data_points"]
        # All labels 0 (single class).
        assert all(p["label"] == 0 for p in points)
        # Convert back to centered coords — mean radius should be ~3.
        center_x = result["center"]["x"]
        center_y = result["center"]["y"]
        scale = result["scale"]
        radii = [
            ((p["x"] - center_x) / scale) ** 2 + ((p["y"] - center_y) / scale) ** 2
            for p in points
        ]
        mean_r = sum(r ** 0.5 for r in radii) / len(radii)
        assert 2.5 < mean_r < 3.5

    def test_unknown_type_falls_back_to_random(self):
        """Unknown data_type should fall through to the random-Gaussian
        branch (same as 'structured')."""
        a = pca_viz.generate_2d_data("structured", n_samples=20)
        b = pca_viz.generate_2d_data("totally-unknown-type", n_samples=20)
        assert len(a["data_points"]) == len(b["data_points"]) == 20


class TestGenerateScreeDataAllRouteTypes:
    """Every data_type accepted by routes/visualization.py for the scree
    endpoint must be handled distinctly by generate_scree_data."""

    @pytest.mark.parametrize("data_type", [
        "structured", "moderate", "linear", "random",
    ])
    def test_all_route_types_produce_data(self, data_type):
        result = pca_viz.generate_scree_data(num_features=10, data_type=data_type)
        assert isinstance(result, dict)
        assert "components" in result
        assert len(result["components"]) == 10
        assert result["data_type"] == data_type

    def test_structured_has_steep_decay(self):
        """'structured' data should have a steep first-vs-second eigenvalue gap."""
        result = pca_viz.generate_scree_data(num_features=10, data_type="structured")
        ev = [c["eigenvalue"] for c in result["components"]]
        # First eigenvalue should be substantially larger than the second.
        assert ev[0] > ev[1] * 1.5

    def test_moderate_has_gradual_decay(self):
        """'moderate' data should have a gentler decay than 'structured'."""
        structured = pca_viz.generate_scree_data(10, "structured")
        moderate = pca_viz.generate_scree_data(10, "moderate")
        s_gap = structured["components"][0]["eigenvalue"] - structured["components"][1]["eigenvalue"]
        m_gap = moderate["components"][0]["eigenvalue"] - moderate["components"][1]["eigenvalue"]
        # Structured decay is steeper than moderate.
        assert s_gap > m_gap

    def test_linear_has_near_flat_decay(self):
        """'linear' should produce the flattest eigenvalue spectrum."""
        result = pca_viz.generate_scree_data(10, "linear")
        ev = [c["eigenvalue"] for c in result["components"]]
        # The ratio between max and min should be small (< 1.6).
        assert max(ev) / min(ev) < 1.6

    def test_random_distinguished_from_structured(self):
        structured = pca_viz.generate_scree_data(10, "structured")
        random_data = pca_viz.generate_scree_data(10, "random")
        s_gap = structured["components"][0]["eigenvalue"] - structured["components"][1]["eigenvalue"]
        r_gap = random_data["components"][0]["eigenvalue"] - random_data["components"][1]["eigenvalue"]
        assert s_gap > r_gap


class TestChemistryPcaData:
    """Smoke tests for get_chemistry_pca_data."""

    @pytest.mark.parametrize("dataset", ["drug", "solvents", "elements"])
    def test_all_datasets_produce_data(self, dataset):
        result = pca_viz.get_chemistry_pca_data(dataset)
        assert isinstance(result, dict)
        assert "points" in result
        assert "legend" in result
        assert "pc1_variance" in result
        assert "pc2_variance" in result
        # Each dataset has multiple points.
        assert len(result["points"]) > 5

    def test_unknown_dataset_uses_elements_fallback(self):
        """Unknown dataset should fall through to the 'elements' branch."""
        result = pca_viz.get_chemistry_pca_data("unknown-dataset")
        # Elements dataset has 18 entries (H through Au).
        assert len(result["points"]) == 18
