"""
RAL Ligand Similarity Engine

Uses the 8 DFT-computed electronic descriptors from Grand_Data as a
feature space to find ligands with similar electronic profiles.

Dimensionality reduction strategy (automatic fallback chain):
  1. **UMAP** (umap-learn) — non-linear, preserves local neighbourhood
     structure, excellent for small-to-medium chemical datasets.
     Better at capturing non-linear structure across the 7 ligand
     families (Phen, Bipy, PyrOx, PyrIm, PyCam, BiOX, BiIM).
  2. **PCA** (sklearn) — linear baseline.  Used when umap-learn is
     not installed, or as a supplementary analysis layer.

Similarity search:
  - Primary:   Euclidean distance in the UMAP/PCA embedding space (KNN).
  - Secondary: Cosine similarity in the original 8D standardised space.

Optional clustering:
  - KMeans on the embedding space for grouping ligands into
    electronic families.

This module is imported lazily by ral_rag.py so that the heavy
numpy/sklearn/umap import only fires when the RAL RAG service is
actually used.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# The 8 electronic descriptor columns used as features
FEATURE_COLS = [
    'HOMO_eV', 'LUMO_eV', 'Gap_eV', 'omega_eV',
    'I_min_eV', 'V_min_eV', 'R1_HOMA', 'R2_HOMA',
]

# Human-readable feature names for explanations / reports
FEATURE_LABELS = [
    'HOMO (eV)', 'LUMO (eV)', 'Gap (eV)', '\u03c9 (eV)',
    'I_min (eV)', 'V_min (eV)', 'R1-HOMA', 'R2-HOMA',
]


@dataclass
class SimilarityResult:
    """A ligand similarity match."""
    name: str
    ligand_class: str
    distance: float              # Euclidean distance in embedding space
    cosine_similarity: float     # cosine similarity in original 8D space
    embedding_coords: List[float]  # coordinates in UMAP / PCA space
    pca_coords: List[float]      # backward-compatible alias for embedding_coords
    features: Dict[str, float]   # original 8 descriptors


@dataclass
class ClusterInfo:
    """Summary of a ligand cluster."""
    cluster_id: int
    ligand_count: int
    ligand_names: List[str]
    class_distribution: Dict[str, int]
    centroid_features: Dict[str, float]


class LigandSimilarityEngine:
    """
    Computes ligand similarity based on electronic descriptors.

    Pipeline:
      1. Extract 8D feature vectors from ligand data.
      2. Standardise (z-score) each feature.
      3. Fit UMAP (preferred) or PCA (fallback) for dimensionality
         reduction and noise filtering.
      4. For similarity: KNN distance in the embedding space, plus
         cosine similarity in the original standardised space.
      5. Optional: fit KMeans for cluster-based grouping.

    Why UMAP over PCA for this domain:
      - The 8 DFT descriptors have non-linear correlations across
        the 7 ligand families (e.g. HOMO-LUMO gap interacts
        differently per class).
      - UMAP preserves *local* neighbourhood structure, which is
        exactly what "find similar ligands" requires.
      - UMAP gives cleaner 2D/3D projections for visualisation,
        better separating the 7 class families.
      - 238 samples with 8 features is trivial for UMAP compute.
    """

    # UMAP default hyperparameters (tuned for 238 ligands, 8 features)
    UMAP_DEFAULTS = {
        'n_neighbors': 15,       # balances local vs global (sqrt(238) ~ 15)
        'min_dist': 0.1,         # tight clusters for clear separation
        'n_components': 3,       # 3D embedding (project to 2D for viz)
        'metric': 'cosine',      # cosine distance suits electronic profiles
        'random_state': 42,
    }

    def __init__(self,
                 n_components: Optional[int] = None,
                 variance_threshold: float = 0.90,
                 force_method: Optional[str] = 'pca'):
        """
        Args:
            n_components: Embedding dimension.  If None, auto-select
                (3 for UMAP, or PCA auto-select by variance threshold).
            variance_threshold: Min cumulative variance for PCA
                (only used when PCA is active and n_components is None).
            force_method: Override auto-detection.  One of 'umap',
                'pca', or None (auto-detect: prefer UMAP).
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.force_method = force_method

        self._fitted = False
        self._method: str = 'none'  # 'umap' | 'pca' | 'none'

        # Ligand metadata
        self._ligand_names: List[str] = []
        self._ligand_classes: List[str] = []

        # Standardisation (shared by both methods)
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None

        # Embedding (UMAP or PCA — the active space)
        self._X_embed: Optional[np.ndarray] = None
        self._n_components_used: int = 0

        # PCA-specific fields (populated when PCA is used)
        self._pca_components: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._explained_variance: Optional[np.ndarray] = None

        # UMAP-specific fields
        self._umap_model = None

        # Raw features
        self._X_raw: Optional[np.ndarray] = None

        # Clustering
        self._cluster_labels: Optional[np.ndarray] = None
        self._n_clusters: int = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, ligands: List[Dict[str, Any]]) -> bool:
        """
        Fit the similarity engine on ligand data.

        Automatically selects UMAP (preferred) or PCA (fallback) based
        on package availability.

        Args:
            ligands: List of dicts, each containing the 8 feature columns.
        """
        if len(ligands) < 3:
            logger.warning("Too few ligands for meaningful similarity")
            return False

        # --- Extract feature matrix ---
        self._ligand_names = [l['name'] for l in ligands]
        self._ligand_classes = [l['class'] for l in ligands]
        X = np.array(
            [[l.get(col, 0.0) for col in FEATURE_COLS] for l in ligands],
            dtype=np.float64,
        )
        self._X_raw = X

        # --- Standardise ---
        self._scaler_mean = X.mean(axis=0)
        self._scaler_std = X.std(axis=0)
        self._scaler_std[self._scaler_std == 0] = 1.0
        X_std = (X - self._scaler_mean) / self._scaler_std

        # --- Choose method ---
        method = self._pick_method()
        self._method = method

        if method == 'umap':
            ok = self._fit_umap(X_std)
        elif method == 'pca':
            ok = self._fit_pca(X_std)
        else:
            logger.error("No dimensionality reduction method available")
            return False

        if not ok:
            return False

        logger.info(
            f"LigandSimilarityEngine fitted: {len(ligands)} ligands, "
            f"method={self._method}, "
            f"{self._n_components_used} components"
        )

        self._fitted = True
        return True

    def _pick_method(self) -> str:
        """Determine which dimensionality reduction method to use.

        Defaults to PCA because umap-learn (via numba/LLVM) can
        segfault in some environments.  Set force_method='umap' to
        opt in.
        """
        if self.force_method:
            return self.force_method

        # Default to PCA — safe, no native code risk
        try:
            import sklearn.decomposition  # noqa: F401
            return 'pca'
        except ImportError:
            pass

        # UMAP only if user explicitly opted in via force_method
        return 'none'

    def _fit_umap(self, X_std: np.ndarray) -> bool:
        """Fit UMAP on standardised features."""
        try:
            import umap
        except ImportError:
            return False

        n_comp = self.n_components or self.UMAP_DEFAULTS['n_components']
        n_comp = min(n_comp, X_std.shape[0] - 1, X_std.shape[1])

        params = dict(self.UMAP_DEFAULTS)
        params['n_components'] = n_comp

        logger.info(
            f"Fitting UMAP: n_neighbors={params['n_neighbors']}, "
            f"min_dist={params['min_dist']}, "
            f"n_components={n_comp}, metric={params['metric']}"
        )

        reducer = umap.UMAP(**params)
        self._X_embed = reducer.fit_transform(X_std)
        self._umap_model = reducer
        self._n_components_used = self._X_embed.shape[1]

        # Also compute PCA for loadings / variance analysis
        self._fit_pca_for_analysis(X_std)

        return True

    def _fit_pca(self, X_std: np.ndarray) -> bool:
        """Fit PCA on standardised features (primary method or analysis supplement)."""
        try:
            import sklearn.decomposition
        except ImportError:
            return False

        if self.n_components is not None:
            n_comp = min(self.n_components, X_std.shape[1], X_std.shape[0])
        else:
            n_comp = min(X_std.shape[1], X_std.shape[0])

        pca = sklearn.decomposition.PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_std)

        # Auto-trim by variance threshold
        if self.n_components is None:
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_keep = int(np.searchsorted(cumvar, self.variance_threshold) + 1)
            n_keep = max(n_keep, 2)
            n_keep = min(n_keep, n_comp)
            if n_keep < n_comp:
                pca = sklearn.decomposition.PCA(n_components=n_keep)
                X_pca = pca.fit_transform(X_std)

        self._pca_components = pca.components_
        self._pca_mean = pca.mean_
        self._explained_variance = pca.explained_variance_ratio_
        self._n_components_used = X_pca.shape[1]

        # If PCA is the primary method, set embedding to PCA space
        if self._method == 'pca':
            self._X_embed = X_pca

        return True

    def _fit_pca_for_analysis(self, X_std: np.ndarray) -> bool:
        """Fit PCA as a supplementary analysis tool (when UMAP is primary)."""
        try:
            import sklearn.decomposition
        except ImportError:
            return False

        n_comp = min(X_std.shape[1], X_std.shape[0])
        pca = sklearn.decomposition.PCA(n_components=n_comp)
        pca.fit(X_std)

        self._pca_components = pca.components_
        self._pca_mean = pca.mean_
        self._explained_variance = pca.explained_variance_ratio_

        return True

    def fit_clustering(self, n_clusters: int = None) -> bool:
        """
        Fit KMeans clustering on the embedding space.

        Args:
            n_clusters: Number of clusters.  If None, auto-select using
                a simple heuristic (sqrt of n_ligands / 3, min 3, max 10).
        """
        if not self._fitted:
            return False

        try:
            import sklearn.cluster
        except ImportError:
            return False

        n = len(self._ligand_names)
        if n_clusters is None:
            n_clusters = max(3, min(10, int(np.sqrt(n / 3))))

        self._n_clusters = n_clusters
        km = sklearn.cluster.KMeans(
            n_clusters=n_clusters, n_init=10, random_state=42
        )
        self._cluster_labels = km.fit_predict(self._X_embed)

        logger.info(
            f"KMeans fitted on {self._method} space: "
            f"{n_clusters} clusters"
        )
        return True

    # ------------------------------------------------------------------
    # Method info
    # ------------------------------------------------------------------

    def get_method_info(self) -> Dict[str, Any]:
        """Return information about which method is being used."""
        info = {
            'method': self._method,
            'fitted': self._fitted,
            'n_ligands': len(self._ligand_names),
            'n_features': len(FEATURE_COLS),
            'n_components': self._n_components_used,
            'n_clusters': self._n_clusters if self._cluster_labels is not None else 0,
        }

        if self._method == 'umap' and self._umap_model is not None:
            info['umap_params'] = {
                'n_neighbors': self._umap_model.n_neighbors,
                'min_dist': self._umap_model.min_dist,
                'metric': self._umap_model.metric,
            }

        if self._explained_variance is not None:
            info['pca_variance_explained'] = [
                round(float(v), 4) for v in self._explained_variance
            ]
            info['pca_total_variance'] = round(
                float(sum(self._explained_variance)), 4
            )

        return info

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def find_similar(self, ligand_name: str, k: int = 5,
                     exclude_same_name: bool = True) -> List[SimilarityResult]:
        """
        Find the k most similar ligands to a given ligand.

        Uses Euclidean distance in the embedding space (UMAP or PCA)
        as primary ranking, with cosine similarity in the original
        8D standardised space as supplementary information.
        """
        if not self._fitted:
            return []

        # Find the index of the query ligand
        query_idx = self._find_ligand_index(ligand_name)
        if query_idx is None:
            return []

        query_embed = self._X_embed[query_idx]
        query_raw = self._X_raw[query_idx]

        # Compute Euclidean distances in embedding space
        distances = np.linalg.norm(self._X_embed - query_embed, axis=1)

        # Compute cosine similarity in original (standardised) space
        cosine_sims = self._compute_cosine_similarities(query_idx)

        # Sort by embedding distance
        indices = np.argsort(distances)

        results = []
        for idx in indices:
            if exclude_same_name and idx == query_idx:
                continue
            if len(results) >= k:
                break

            features = {
                FEATURE_LABELS[i]: float(self._X_raw[idx][i])
                for i in range(len(FEATURE_COLS))
            }
            coords = self._X_embed[idx].tolist()
            results.append(SimilarityResult(
                name=self._ligand_names[idx],
                ligand_class=self._ligand_classes[idx],
                distance=float(distances[idx]),
                cosine_similarity=float(cosine_sims[idx]),
                embedding_coords=coords,
                pca_coords=coords,  # backward-compatible
                features=features,
            ))

        return results

    def find_similar_by_features(self, feature_dict: Dict[str, float],
                                  k: int = 5) -> List[SimilarityResult]:
        """
        Find similar ligands given a set of electronic descriptor values.

        Useful when the "reference" is not in the database (e.g., a
        user-specified ideal property profile).
        """
        if not self._fitted:
            return []

        # Build query vector
        try:
            query_raw = np.array(
                [feature_dict.get(col, 0.0) for col in FEATURE_COLS],
                dtype=np.float64
            )
        except (KeyError, TypeError):
            return []

        # Transform through standardisation
        query_std = (query_raw - self._scaler_mean) / self._scaler_std

        # Transform to embedding space
        if self._method == 'umap' and self._umap_model is not None:
            query_embed = self._umap_model.transform(query_std.reshape(1, -1))[0]
        elif self._pca_components is not None:
            query_embed = query_std @ self._pca_components.T
        else:
            return []

        distances = np.linalg.norm(self._X_embed - query_embed, axis=1)
        indices = np.argsort(distances)[:k]

        results = []
        for idx in indices:
            features = {
                FEATURE_LABELS[i]: float(self._X_raw[idx][i])
                for i in range(len(FEATURE_COLS))
            }
            coords = self._X_embed[idx].tolist()
            results.append(SimilarityResult(
                name=self._ligand_names[idx],
                ligand_class=self._ligand_classes[idx],
                distance=float(distances[idx]),
                cosine_similarity=0.0,
                embedding_coords=coords,
                pca_coords=coords,
                features=features,
            ))

        return results

    def recommend_for_ligand(self, ligand_name: str, k: int = 5,
                              same_class_only: bool = False,
                              different_class_only: bool = False
                              ) -> List[SimilarityResult]:
        """
        Recommend alternative ligands similar to a given one.

        This is the key method for the RAG "suggest similar ligand"
        use case.  It finds the k nearest neighbours in the embedding
        space and returns them with an explanation-ready data structure.

        Args:
            ligand_name: Name of the reference ligand.
            k: Number of recommendations.
            same_class_only: Only return ligands from the same class.
            different_class_only: Only return ligands from a different class.
        """
        results = self.find_similar(ligand_name, k=k * 3)  # over-fetch
        if not results:
            return []

        query_idx = self._find_ligand_index(ligand_name)
        target_class = (
            self._ligand_classes[query_idx]
            if query_idx is not None else ''
        )

        if same_class_only:
            results = [r for r in results if r.ligand_class == target_class]

        if different_class_only:
            results = [r for r in results if r.ligand_class != target_class]

        return results[:k]

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def get_cluster_for_ligand(self, ligand_name: str) -> Optional[int]:
        """Get the cluster ID for a specific ligand."""
        if self._cluster_labels is None:
            return None
        idx = self._find_ligand_index(ligand_name)
        if idx is None:
            return None
        return int(self._cluster_labels[idx])

    def get_cluster_members(self, cluster_id: int) -> List[Dict[str, Any]]:
        """Get all ligands in a given cluster."""
        if self._cluster_labels is None:
            return []
        members = []
        for i, label in enumerate(self._cluster_labels):
            if int(label) == cluster_id:
                members.append({
                    'name': self._ligand_names[i],
                    'class': self._ligand_classes[i],
                    'embedding_coords': self._X_embed[i].tolist(),
                })
        return members

    def get_all_clusters(self) -> List[ClusterInfo]:
        """Get summary info for all clusters."""
        if self._cluster_labels is None:
            return []

        clusters: Dict[int, List[int]] = {}
        for i, label in enumerate(self._cluster_labels):
            clusters.setdefault(int(label), []).append(i)

        result = []
        for cid, indices in sorted(clusters.items()):
            class_dist: Dict[str, int] = {}
            for idx in indices:
                cls = self._ligand_classes[idx]
                class_dist[cls] = class_dist.get(cls, 0) + 1

            centroid = self._X_raw[indices].mean(axis=0)
            centroid_dict = {
                FEATURE_LABELS[j]: round(float(centroid[j]), 3)
                for j in range(len(FEATURE_COLS))
            }

            result.append(ClusterInfo(
                cluster_id=cid,
                ligand_count=len(indices),
                ligand_names=[self._ligand_names[i] for i in indices],
                class_distribution=class_dist,
                centroid_features=centroid_dict,
            ))

        return result

    # ------------------------------------------------------------------
    # Embedding / visualisation data
    # ------------------------------------------------------------------

    def get_embedding_data(self) -> Dict[str, Any]:
        """
        Return embedding projections of all ligands for visualisation.

        Works with both UMAP and PCA.  Returns 2D coordinates
        (first two components) plus class labels and cluster IDs.
        """
        if not self._fitted:
            return {}

        points = []
        for i in range(len(self._ligand_names)):
            point = {
                'name': self._ligand_names[i],
                'class': self._ligand_classes[i],
                'x': float(self._X_embed[i][0]),
                'y': float(self._X_embed[i][1]) if self._n_components_used > 1 else 0.0,
            }
            if self._n_components_used > 2:
                point['z'] = float(self._X_embed[i][2])
            if self._cluster_labels is not None:
                point['cluster'] = int(self._cluster_labels[i])
            points.append(point)

        return {
            'points': points,
            'method': self._method,
            'n_components': self._n_components_used,
            'n_ligands': len(self._ligand_names),
            'n_clusters': self._n_clusters if self._cluster_labels is not None else 0,
            'feature_labels': FEATURE_LABELS,
            'explained_variance_ratio': (
                [round(float(v), 4) for v in self._explained_variance]
                if self._explained_variance is not None else None
            ),
        }

    def get_pca_data(self) -> Dict[str, Any]:
        """Backward-compatible alias for get_embedding_data()."""
        return self.get_embedding_data()

    def get_pca_loadings(self) -> Dict[str, Any]:
        """
        Return PCA loadings (correlation of original features with PCs).

        Useful for understanding which descriptors drive the clustering.
        Always uses PCA (even when UMAP is the primary embedding method).
        """
        if not self._fitted:
            return {}
        if self._pca_components is None:
            return {}

        X_std = (self._X_raw - self._scaler_mean) / self._scaler_std

        # Project onto PCA components
        X_pca = X_std @ self._pca_components.T

        loadings = []
        n_pcs = min(self._n_components_used, X_pca.shape[1])
        for pc_idx in range(n_pcs):
            pc_scores = X_pca[:, pc_idx]
            correlations = []
            for feat_idx in range(len(FEATURE_COLS)):
                corr = np.corrcoef(X_std[:, feat_idx], pc_scores)[0, 1]
                correlations.append(round(float(corr), 3))
            loadings.append({
                'pc': f'PC{pc_idx + 1}',
                'variance_explained': round(
                    float(self._explained_variance[pc_idx]), 4
                ) if self._explained_variance is not None else None,
                'loadings': dict(zip(FEATURE_LABELS, correlations)),
            })

        return {
            'loadings': loadings,
            'feature_labels': FEATURE_LABELS,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_ligand_index(self, ligand_name: str) -> Optional[int]:
        """Find a ligand index by name (case-insensitive, with substring fallback)."""
        name_lower = ligand_name.lower()

        # Exact match
        for i, n in enumerate(self._ligand_names):
            if n.lower() == name_lower:
                return i

        # Substring match
        for i, n in enumerate(self._ligand_names):
            if name_lower in n.lower() or n.lower() in name_lower:
                return i

        return None

    def _compute_cosine_similarities(self, query_idx: int) -> np.ndarray:
        """Compute cosine similarity between query and all ligands in original space."""
        query_std = (
            (self._X_raw[query_idx] - self._scaler_mean) / self._scaler_std
        )
        all_std = (self._X_raw - self._scaler_mean) / self._scaler_std

        norms = np.linalg.norm(all_std, axis=1)
        norms[norms == 0] = 1.0
        query_norm = np.linalg.norm(query_std)
        if query_norm == 0:
            query_norm = 1.0

        return all_std @ query_std / (norms * query_norm)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_similarity_engine: Optional[LigandSimilarityEngine] = None


def get_similarity_engine() -> LigandSimilarityEngine:
    """Get or create the global similarity engine, fitting on RAL database."""
    global _similarity_engine
    if _similarity_engine is not None and _similarity_engine._fitted:
        return _similarity_engine

    from .ral_database import get_ral_database

    db = get_ral_database()
    if not db._loaded:
        if not db.load():
            logger.error("Cannot fit similarity engine: RAL database not loaded")
            return LigandSimilarityEngine()  # return unfitted

    # Collect all ligands from all classes
    all_ligands = []
    for cls in db._ligands_by_class:
        all_ligands.extend(db.get_ligands_by_class(cls))

    if not all_ligands:
        return LigandSimilarityEngine()

    _similarity_engine = LigandSimilarityEngine()
    _similarity_engine.fit(all_ligands)
    _similarity_engine.fit_clustering()

    return _similarity_engine