"""
RedCross (Reductive Coupling) Database Service

Loads and indexes two complementary JSON datasets:
  1. ligand_database.json  — 238 ligands with DFT-computed electronic descriptors
     across 7 classes (Phen, Bpy, PyrOx, PyrIm, PyCam, BiOX, BiIM).
  2. reaction_database.json  — 225 fully-curated reductive coupling reactions
     with DOI, title, optimum ligand, coupling partner, and ligand knowledge.

JSON column schemas:
  ligand_database.json:
    Ligand Name, Ligand Abbreviation, Class,
    HOMO (eV), LUMO (eV), Gap (eV), ω (eV),
    I_min (eV), V_min (eV), R1-HOMA, R2-HOMA

  reaction_database.json:
    DOI, Title, Optimum Ligand, Reaction, Ligand Knowledge
"""

import os
import re
import json
import math
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# LigandClassifier — replaces CLASS_ALIASES + LIGAND_NAME_ALIASES
# ═══════════════════════════════════════════════════════════════════════════

# The 7 canonical class codes in the database
VALID_CLASSES: Tuple[str, ...] = (
    'Phen', 'Bpy', 'PyrOx', 'PyrIm', 'PyCam', 'BiOX', 'BiIM',
)


class ClassificationMethod(Enum):
    """Tracks HOW a class was determined — useful for debugging."""
    CANONICAL_EXACT = 'canonical_exact'       # exact canonical class name found
    BRACKET_ABBREV = 'bracket_abbrev'         # abbreviation inside [...]
    SCAFFOLD_RULE = 'scaffold_rule'           # structural scaffold keyword
    ABBREV_PATTERN = 'abbrev_pattern'         # short code (bpy, phen, etc.)
    LIGAND_DB_MATCH = 'ligand_db_match'       # matched against loaded DB names
    FALLBACK_NONE = 'fallback_none'           # no match → unknown


@dataclass
class ClassMatch:
    """Result of classifying a single ligand segment."""
    cls: str                         # one of VALID_CLASSES or 'unknown'
    method: ClassificationMethod
    matched_term: str                # the text that triggered this class
    confidence: int                  # 1-5 (higher = more specific match)


@dataclass
class ClassificationResult:
    """Aggregated result of classifying a (potentially multi-ligand) string."""
    primary_class: str               # first/most confident class
    all_classes: List[str]           # all distinct classes found (deduped, order-preserving)
    matches: List[ClassMatch]        # every individual match
    cleaned_name: str                # the primary name after annotation stripping
    parsed_refs: List['ParsedLigandRef'] = field(default_factory=list)  # structured parse


@dataclass
class ParsedLigandRef:
    """Structured representation of a single ligand from Optimum Ligand field.

    Canonical format: "full_name [abbreviation] explanation"
    Multiple ligands in one field are separated by ";".
    Variations: parenthetical abbreviation (bipy), no brackets at all,
    "+" separator for dual-ligand systems.
    """
    full_name: str          # IUPAC / descriptive name (before the bracket)
    abbreviation: str       # short code extracted from [...] or (...)
    explanation: str        # usage notes, preformed complex info, etc.
    raw_segment: str        # original text for this single ligand


class ReactionLigandIndexer:
    """
    Builds a deterministic lookup index from reaction database entries.

    At load time, parses ALL 225 reaction Optimum Ligand strings using
    the format grammar ("name [abbreviation] explanation") and
    cross-references against the ligand database to learn new
    abbreviation → class mappings that supplement the hardcoded table.

    Three-layer index architecture:
      Layer 1 (hardcoded): Known chemical abbreviations (bpy, phen, dtbbpy, …)
      Layer 2 (ligand DB):  All 238 ligand abbreviations from ligand_database.json
      Layer 3 (reaction DB): Abbreviations extracted from 152 unique Optimum
                             Ligand entries, validated by cross-referencing full
                             names against the ligand database.

    Also builds a ``raw_text_to_class`` index that maps the RAW (unstripped)
    segment text to its resolved class — used as a fast-path cache so
    that reactions whose Optimum Ligand was successfully parsed at load
    time never need to re-run the full scaffold pipeline.
    """

    # Canonical class names embedded in descriptions like "specific BiIM ligand"
    _CLASS_NAME_PATTERN = re.compile(
        r'\b(' + '|'.join(re.escape(c) for c in VALID_CLASSES) + r')\b', re.I
    )

    def __init__(self):
        self._raw_to_class: Dict[str, str] = {}
        self._abbrev_to_class: Dict[str, str] = {}
        self._stats: Dict[str, int] = {}

    def build(self, reaction_records: List[Dict[str, str]],
              ligand_name_to_class: Dict[str, str],
              ligand_abbr_to_class: Dict[str, str],
              classifier: 'LigandClassifier') -> None:
        """Parse all reaction entries and build derived indices.

        Parameters
        ----------
        reaction_records : list of dict
            Each dict has 'Optimum Ligand' key.
        ligand_name_to_class : dict
            Lowercase full ligand name → canonical class (from ligand DB).
        ligand_abbr_to_class : dict
            Lowercase abbreviation → canonical class (from ligand DB).
        classifier : LigandClassifier
            Used for parsing and scaffold matching.
        """
        parsed_total = 0
        learned_abbrs = 0
        raw_hits = 0

        for rec in reaction_records:
            raw_opt = str(rec.get('Optimum Ligand') or '').strip()
            if not raw_opt:
                continue

            parsed_total += 1
            refs = classifier.parse_optimum_ligand(raw_opt)

            for ref in refs:
                # --- Fast-path: if bracket abbrev is in an existing index ---
                resolved_class = None
                if ref.abbreviation:
                    abbr_key = LigandClassifier._prime_normalize(
                        ref.abbreviation.lower().strip()
                    )
                    # Check all three layers
                    if abbr_key in classifier._abbrev_index:
                        resolved_class = classifier._abbrev_index[abbr_key]
                    # Check for embedded class name ("specific BiIM ligand")
                    if resolved_class is None:
                        cn_match = self._CLASS_NAME_PATTERN.search(ref.abbreviation)
                        if cn_match:
                            resolved_class = cn_match.group(1)
                            # Capitalise to match VALID_CLASSES casing
                            for vc in VALID_CLASSES:
                                if vc.lower() == resolved_class.lower():
                                    resolved_class = vc
                                    break
                    # Try compound abbrev decomposition:
                    # "Pent(3,3)-Bis(IndOx)" → extract "Pent", "IndOx" sub-tokens
                    if resolved_class is None:
                        resolved_class = self._decompose_compound_abbrev(
                            ref.abbreviation, classifier
                        )

                # --- Fallback: match full_name against ligand DB ---
                if resolved_class is None and ref.full_name:
                    name_norm = LigandClassifier._prime_normalize(
                        ref.full_name.lower().strip()
                    )
                    if name_norm in ligand_name_to_class:
                        resolved_class = ligand_name_to_class[name_norm]

                # --- Fallback: scaffold rules on full_name ---
                if resolved_class is None and ref.full_name:
                    scaffold_match = classifier._classify_by_scaffold(ref.full_name)
                    if scaffold_match and scaffold_match.cls in VALID_CLASSES:
                        resolved_class = scaffold_match.cls

                # --- Record resolved class for this raw segment ---
                if resolved_class and resolved_class in VALID_CLASSES:
                    raw_hits += 1
                    self._raw_to_class[ref.raw_segment] = resolved_class

                    # Learn abbreviation → class mapping if not already known
                    if ref.abbreviation:
                        abbr_key = LigandClassifier._prime_normalize(
                            ref.abbreviation.lower().strip()
                        )
                        if abbr_key not in classifier._abbrev_index:
                            classifier._abbrev_index[abbr_key] = resolved_class
                            learned_abbrs += 1

        self._stats = {
            'parsed_reactions': parsed_total,
            'raw_segment_hits': raw_hits,
            'learned_abbrevs': learned_abbrs,
            'raw_index_size': len(self._raw_to_class),
            'abbrev_index_total': len(classifier._abbrev_index),
        }
        logger.debug(
            "ReactionLigandIndexer built: %s", self._stats
        )

    def _decompose_compound_abbrev(
        self, abbrev: str, classifier: 'LigandClassifier'
    ) -> Optional[str]:
        """Try to extract a class from compound abbreviations.

        Handles patterns like:
          - ``Pent(3,3)-Bis(IndOx)`` → extract "Pent", "IndOx"
          - ``CycP(1,1)-Bis(IndOx)`` → extract "CycP", "IndOx"
          - ``R,R-Ph-BOXiPr`` → extract "BOX"
        """
        # Split on hyphens and parentheses to get sub-tokens
        # Remove parenthetical groups first, but keep the text before them
        cleaned = re.sub(r'\([^)]*\)', '', abbrev)
        parts = re.split(r'[-,/\s]+', cleaned)

        for part in parts:
            part = part.strip()
            if len(part) < 2:
                continue
            key = part.lower()
            if key in classifier._abbrev_index:
                cls = classifier._abbrev_index[key]
                if cls in VALID_CLASSES:
                    return cls
            # Also try embedded class names
            cn = self._CLASS_NAME_PATTERN.search(part)
            if cn:
                for vc in VALID_CLASSES:
                    if vc.lower() == cn.group(1).lower():
                        return vc

        # If no sub-token matched, try scaffold rules on the full abbrev
        # (without parenthetical groups that look like position specifiers)
        m = classifier._classify_by_scaffold(abbrev)
        if m and m.cls in VALID_CLASSES:
            return m.cls

        return None

    def lookup_raw(self, raw_segment: str) -> Optional[str]:
        """Fast-path: look up a raw segment's pre-resolved class."""
        return self._raw_to_class.get(raw_segment)

    def get_stats(self) -> Dict[str, int]:
        """Return build statistics."""
        return dict(self._stats)


class LigandClassifier:
    """
    Rigid, multi-stage ligand classifier pipeline.

    Replaces the fragile CLASS_ALIASES dict (substring ``in`` matching)
    and LIGAND_NAME_ALIASES dict with a deterministic pipeline:

    ┌──────────────────────────────────────────────────────────────┐
    │ Stage 0: Optimum Ligand format parsing                       │
    │   • Parse "name [abbreviation] explanation" structure        │
    │   • Bracket-aware splitting on ';' and '+'                   │
    │   • Smart bracket selection: skip IUPAC [1,2-d], skip        │
    │     complex formulas [NiCl2(bpy)Cl2]                         │
    │   • Produces ParsedLigandRef objects with name/abbr/expl     │
    └──────────────────────┬───────────────────────────────────────┘
                             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ Stage 0.5: ReactionLigandIndexer fast-path                   │
    │   • Pre-computed at load time from all 225 reaction entries  │
    │   • Maps raw Optimum Ligand segments → class (cache hit)    │
    │   • Skips the full pipeline when the answer is already known │
    └──────────────────────┬───────────────────────────────────────┘
                             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ Stage 1: Pre-processing                                      │
    │   • Normalize Unicode primes (′ → ')                        │
    │   • Normalize whitespace                                     │
    └──────────────────────┬───────────────────────────────────────┘
                             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ Stage 2: Segment splitting                                   │
    │   • Split multi-ligand entries on ';' (bracket-aware)        │
    │   • Chemical-structure-aware paren handling                  │
    │   • Each segment is classified independently                  │
    └──────────────────────┬───────────────────────────────────────┘
                             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ Stage 3: Bracket abbreviation extraction                     │
    │   • Extract short codes from [...] (e.g. [BPhen], [tBu-terpy])│
    │   • Handles "specific X ligand" descriptions                │
    │   • Handles compound abbrevs: Pent(3,3)-Bis(IndOx)            │
    │   • Match against 3-layer abbrev index                       │
    │   • These are HIGH CONFIDENCE — authors wrote them explicitly│
    └──────────────────────┬───────────────────────────────────────┘
                             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ Stage 4: Structural scaffold detection (regex, ordered rules)│
    │   • Rules ordered specific → general                        │
    │   • Each rule targets a UNIQUE chemical scaffold              │
    │   • First match wins per segment                             │
    │   • No substring ``in`` — all use word-boundary-aware regex  │
    └──────────────────────┬───────────────────────────────────────┘
                             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ Stage 5: Abbreviation pattern matching                       │
    │   • Short isolated codes: dtbbpy, phen, pybox, etc.          │
    │   • Requires word-boundary or start/end anchoring            │
    │   • Catches shorthand not covered by scaffold rules           │
    └──────────────────────┬───────────────────────────────────────┘
                             ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ Stage 6: Resolution                                           │
    │   • Merge per-segment results                                │
    │   • Primary class = most confident / first specific match    │
    │   • Fallback to 'unknown' if nothing matched                 │
    └──────────────────────────────────────────────────────────────┘

    The pipeline is supplemented by ReactionLigandIndexer, which at
    load time parses all 225 reactions and pre-resolves their classes,
    building a 3-layer abbreviation index (hardcoded + ligand-DB +
    reaction-DB derived).
    """

    def __init__(self):
        # Instance-level abbreviation index: starts with hardcoded defaults,
        # extended at runtime by build_abbrev_index() from ligand_database.
        self._abbrev_index: Dict[str, str] = dict(self.ABBREV_TO_CLASS)
        # Reaction-derived indexer — set by RedCrossDatabase.load() after
        # both JSON files are loaded.  Provides a fast-path cache that
        # maps raw Optimum Ligand segments to their pre-resolved class.
        self.reaction_indexer: Optional[ReactionLigandIndexer] = None

    # ------------------------------------------------------------------
    # Abbreviation table — maps explicit short codes to canonical class
    # These are extracted from bracket annotations like [BPhen], [PyBox]
    # ------------------------------------------------------------------
    ABBREV_TO_CLASS: Dict[str, str] = {
        # Bipyridine
        'bpy': 'Bpy', 'bipy': 'Bpy',
        'dtbbpy': 'Bpy', 'dtbpy': 'Bpy',
        "4,4'-tbu-bpy": 'Bpy', '4,4-tbu-bpy': 'Bpy',
        "4,4'-och3-bpy": 'Bpy', "4,4'-oCH3-bpy": 'Bpy', "4,4-oCH3-bpy": 'Bpy',
        "4,4'-ch3-bpy": 'Bpy', "4,4'-cH3-bpy": 'Bpy',
        "5,5'-ch3-bpy": 'Bpy', "5,5'-cH3-bpy": 'Bpy',
        "6,6'-ch3-bpy": 'Bpy', "6,6'-cH3-bpy": 'Bpy',
        "4,4'-cf3-bpy": 'Bpy',
        "4,4'-nr2-bpy": 'Bpy', "4,4'-nme2-bpy": 'Bpy',
        "4,4'-ph-bpy": 'Bpy',
        "4,4'-coch3-bpy": 'Bpy', "4,4'-co2ch3-bpy": 'Bpy',
        # Phenanthroline
        'phen': 'Phen',
        'bphen': 'Phen', 'bathophenanthroline': 'Phen',
        'bcp': 'Phen', 'bathocuproine': 'Phen', 'bcpe': 'Phen',
        'neocuproine': 'Phen', 'dmphen': 'Phen',
        'tmphen': 'Phen', 'me4phen': 'Phen',
        "4,7-och3-phen": 'Phen', "4,7-oCH3-phen": 'Phen',
        "4,7-och3-2,9-ch3-phen": 'Phen',
        "(meo)2phen": 'Phen',
        # PyrOx
        'pyrox': 'PyrOx', 'pyox': 'PyrOx', 'pyoxime': 'PyrOx',
        "pyr-indox": 'PyrOx',
        "4-och3-pybox": 'PyrOx', "4-cl-pybox": 'PyrOx',
        "pyboxipr": 'PyrOx', "pyboxsbu": 'PyrOx', "pybox": 'PyrOx',
        # PyrIm
        'pyrim': 'PyrIm',
        "5-ph-pyrim": 'PyrIm',
        # PyCam
        'pycam': 'PyCam', 'pycamh': 'PyCam', 'pycamcn': 'PyCam',
        'pybcam': 'PyCam', 'pybcamcn': 'PyCam',
        'bpycamcn': 'PyCam', 'bpycam': 'PyCam',
        "4,4'-tbu-bpycamcn": 'PyCam',
        # BiOX
        'biox': 'BiOX', 'box': 'BiOX',
        'bisoxazoline': 'BiOX',
        'cy-biox': 'BiOX',
        "r,r-ph-boxipr": 'BiOX', "(s,s)-4-heptyl-biox": 'BiOX',
        "(s,s)-sbu-biox": 'BiOX',
        "r,r-ph-box": 'BiOX',
        # IndOx fused systems → BiOX
        'pentr': 'BiOX',  # Pent(3,3)-Bis(IndOx)
        'cycp': 'BiOX',  # CycP(1,1)-Bis(IndOx)
        "(s,s)-ph-box": 'BiOX',
        # BiIM
        'biim': 'BiIM',
        "(s,s)-sbu-biim3-tbuph": 'BiIM',
        # Terpyridine (→ Phen)
        'terpy': 'Phen', 'tbu-terpy': 'Phen',
        "4'-oh-terpy": 'Phen', "4'-och3-terpy": 'Phen',
        "4'-p-anisyl-terpy": 'Phen',
        # Pyridine / mono-dentate (→ Bpy)
        'py': 'Bpy', 'dmap': 'Bpy',
        "3-f-py": 'Bpy', "4-cn-py": 'Bpy',
        # Misc
        'bpp': 'Phen', 'mebpp': 'Phen',   # pyrazolyl-pyridine → Phen
        'phox': 'PyrOx',                   # phosphino-oxazoline
        'imdZ': 'Bpy', 'imdz': 'Bpy',     # imidazole → Bpy (mono-dentate)
        '5-cn-imdz': 'Bpy',
        'mebpi': 'Phen',                   # bis(pyridylimine) → Phen
        'ttbtpy': 'Phen',                  # tri-tert-butyl terpy
        # Additional entries discovered from reaction data cross-referencing
        "5,5'-cf3-bpy": 'Bpy',            # 5,5'-bis(trifluoromethyl)-2,2'-bipyridine
        '(s)-sbu-biox': 'BiOX',            # (S)-sec-butyl-BiOX variant
        'picolylamine': 'Bpy',             # di(2-picolyl)amine → pyridine-based
        'dpa': 'Bpy',                      # di(2-picolyl)amine abbreviation
    }

    # ------------------------------------------------------------------
    # Structural scaffold rules
    # Ordered: most specific scaffolds FIRST to avoid premature general match
    # Each rule: (compiled_regex, target_class, confidence, description)
    # ------------------------------------------------------------------
    _SCAFFOLD_RULES: List[Tuple[re.Pattern, str, int, str]] = [
        # --- Bidentate heterocyclic pairs (most specific) ---
        (
            re.compile(
                r'\bbis\s*\(?imida[zt]olin', re.I
            ),
            'BiIM', 5,
            'bis(imidazoline) scaffold',
        ),
        (
            re.compile(
                r'\bdiimida[zt]olin', re.I
            ),
            'BiIM', 5,
            'diimidazoline scaffold',
        ),
        (
            re.compile(
                r'\bbis\s*\(?oxa[zt]olin', re.I
            ),
            'BiOX', 5,
            'bis(oxazoline) scaffold',
        ),
        (
            re.compile(
                r'\b(?:4,5-dihydro-1[hh]?-)?imida[zt]ol-2-yl\)?pyridin', re.I
            ),
            'PyrIm', 5,
            'pyridyl-imidazoline (PyrIm)',
        ),
        (
            re.compile(
                r'\b(?:4,5-dihydro)?oxa[zt]ol-2-yl\)?pyridin', re.I
            ),
            'PyrOx', 5,
            'pyridyl-oxazoline (PyrOx)',
        ),
        # --- Bidentate with pyridine core ---
        (
            re.compile(
                r'\bpyridin.*(?:oxa[zt]olin|pyox)', re.I
            ),
            'PyrOx', 4,
            'pyridine-oxazoline variant',
        ),
        (
            re.compile(
                r'\bpyridin.*(?:imida[zt]olin)', re.I
            ),
            'PyrIm', 4,
            'pyridine-imidazoline variant',
        ),
        (
            re.compile(
                r'\b(?:pyridin|picolin)(?:e|yl)?-?(?:2,6-)?bis'
                r'\(?carboximida', re.I
            ),
            'PyCam', 5,
            'pyridine-bis(carboximidamide) = PyBCam',
        ),
        (
            re.compile(
                r'\b(?:pyridin|picolin)(?:e|yl)?-?carboximida', re.I
            ),
            'PyCam', 4,
            'pyridine-carboxamidine = PyCam',
        ),
        (
            re.compile(
                r'\bpyridin.*carboxamid', re.I
            ),
            'PyCam', 4,
            'pyridine-carboxamide variant',
        ),
        # --- Tridentate ---
        (
            re.compile(
                r'\bterpyridin', re.I
            ),
            'Phen', 4,
            'terpyridine (→ Phen)',
        ),
        (
            re.compile(
                r'\bpyridine-2,6-bis\s*\(?'
                r'(?:4-isopropyl-2-oxa[zt]olin|4-methyl-1[hh]?-pyra[zt]ol)', re.I
            ),
            'PyrOx', 4,
            'PyBox = pyridine-bis(oxazoline)',
        ),
        (
            re.compile(
                r'\b2,6-bis\s*\(?[\w-]*oxa[zt]olin', re.I
            ),
            'BiOX', 4,
            'pyridine-bis(oxazoline) → BiOX family',
        ),
        # --- Bidentate diimine (Phen, Bpy) ---
        (
            re.compile(
                r'\bphenanthrolin', re.I
            ),
            'Phen', 5,
            'phenanthroline scaffold',
        ),
        (
            re.compile(
                r'\bbipyridin', re.I
            ),
            'Bpy', 5,
            'bipyridine scaffold',
        ),
        (
            re.compile(
                r'\bdipyridin', re.I
            ),
            'Bpy', 4,
            'dipyridine variant',
        ),
        # --- Amidines / amidoximes ---
        (
            re.compile(
                r'\bamidoxim', re.I
            ),
            'PyCam', 3,
            'amidoxime (PyCam family)',
        ),
        (
            re.compile(
                r'\boxim', re.I
            ),
            'PyrOx', 3,
            'oxime (PyrOx/PyOxime family)',
        ),
        (
            re.compile(
                r'\bamidin', re.I
            ),
            'PyCam', 3,
            'amidine (PyCam family)',
        ),
        # --- Mono-dentate (fallback) ---
        (
            re.compile(
                r'\bpyridin', re.I
            ),
            'Bpy', 2,
            'pyridine (mono-dentate → Bpy)',
        ),
        # --- Pyridyl variants (not just "pyridin") ---
        (
            re.compile(
                r'\b(?:tri|di|tetra)?pyridyl', re.I
            ),
            'Bpy', 3,
            'pyridyl / tripyridyl / dipicolyl (→ Bpy)',
        ),
        (
            re.compile(
                r'\bpicolylamin', re.I
            ),
            'Bpy', 4,
            'picolylamine / DPA (pyridine-based → Bpy)',
        ),
        # --- Oxazole (not just oxazoline) ---
        (
            re.compile(
                r'\boxazole', re.I
            ),
            'BiOX', 3,
            'oxazole ring system (→ BiOX)',
        ),
        # --- Pyridyl-carboxamidine (PyCam family) ---
        (
            re.compile(
                r'\bpyridyl.*carboxamidin', re.I
            ),
            'PyCam', 4,
            'pyridyl-carboxamidine (→ PyCam)',
        ),
        # --- Indeno-oxazole / indeno-imidazole fused systems ---
        (
            re.compile(
                r'indeno.*oxa[zt]ol', re.I
            ),
            'BiOX', 4,
            'indeno-oxazole fused system (→ BiOX)',
        ),
        # --- Bare heterocycle fallbacks ---
        (
            re.compile(
                r'\boxa[zt]olin', re.I
            ),
            'BiOX', 3,
            'bare oxazoline → BiOX',
        ),
        (
            re.compile(
                r'\bimida[zt]olin', re.I
            ),
            'BiIM', 3,
            'bare imidazoline → BiIM',
        ),
        (
            re.compile(
                r'\bimida[zt]ol', re.I
            ),
            'Bpy', 2,
            'bare imidazole (mono-dentate → Bpy)',
        ),
        (
            re.compile(
                r'\bpyra[zt]olyl', re.I
            ),
            'Phen', 3,
            'pyrazolyl → Phen',
        ),
        # --- Pyrazolyl-pyridine (bpp family) ---
        (
            re.compile(
                r'\bpyrazolyl.*pyridin', re.I
            ),
            'Phen', 4,
            'pyrazolyl-pyridine (→ Phen)',
        ),
        # --- Relaxed fallbacks (NO word boundary) ---
        # These catch scaffold keywords fused to substituent prefixes
        # where \b fails (e.g. "methoxypyridine", "dimethoxypyridyl").
        # They have lower confidence than the word-bounded versions
        # above, so they only fire when the \b versions don't match.
        (
            re.compile(r'pyridin', re.I),
            'Bpy', 1,
            'pyridine fused (no word boundary)',
        ),
        (
            re.compile(r'pyridyl', re.I),
            'Bpy', 1,
            'pyridyl fused (no word boundary)',
        ),
        (
            re.compile(r'carboxamidin', re.I),
            'PyCam', 1,
            'carboxamidine fused (no word boundary)',
        ),
        (
            re.compile(r'amidoxim', re.I),
            'PyCam', 1,
            'amidoxime fused (no word boundary)',
        ),
        (
            re.compile(r'oxazole', re.I),
            'BiOX', 1,
            'oxazole fused (no word boundary)',
        ),
        (
            re.compile(r'oxazolin', re.I),
            'BiOX', 1,
            'oxazoline fused (no word boundary)',
        ),
        (
            re.compile(r'imidazolin', re.I),
            'BiIM', 1,
            'imidazoline fused (no word boundary)',
        ),
    ]

    # ------------------------------------------------------------------
    # Abbreviation patterns — short codes that may appear in running text
    # These are distinct from bracket abbreviations: they can appear inline
    # Word-boundary anchored to prevent false positives
    # ------------------------------------------------------------------
    _ABBREV_PATTERNS: List[Tuple[re.Pattern, str, int]] = [
        (re.compile(r'\bdtbb?py\b', re.I),            'Bpy',  4),
        (re.compile(r'\b(?:bpy|bipy)\b', re.I),       'Bpy',  3),
        (re.compile(r'\b(?:phen|bphen)\b', re.I),     'Phen', 3),
        (re.compile(r'\bbathocuproin\w*\b', re.I),    'Phen', 4),
        (re.compile(r'\bneocuproin\w*\b', re.I),      'Phen', 4),
        (re.compile(r'\b(?:tmphen|me4phen)\b', re.I), 'Phen', 4),
        (re.compile(r'\bpybox\w*\b', re.I),           'PyrOx', 3),
        (re.compile(r'\b(?:pyrox|pyox)\b', re.I),     'PyrOx', 3),
        (re.compile(r'\bpybcam\w*\b', re.I),          'PyCam', 4),
        (re.compile(r'\bpycam\w*\b', re.I),           'PyCam', 3),
        (re.compile(r'\bbiox\w*\b', re.I),            'BiOX',  3),
        (re.compile(r'(?:^|(?<=\s)|(?<=-)|(?<=/))box(?:$|(?=\s)|(?=,)|(?=\)))', re.I), 'BiOX', 3),
        (re.compile(r'\bbisoxazoline\b', re.I), 'BiOX', 3),
        (re.compile(r'\bbiim\w*\b', re.I),            'BiIM',  3),
        (re.compile(r'\bterpy\w*\b', re.I),           'Phen', 3),
        (re.compile(r'\bbpp\w*\b', re.I),             'Phen', 3),
        (re.compile(r'\bdmap\b', re.I),               'Bpy',  4),
        (re.compile(r'\bpyridine\b', re.I),           'Bpy',  2),
    ]

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(s: str) -> str:
        """Stage 1: Unicode and whitespace normalization."""
        s = s.replace('\u2032', "'").replace('\u2019', "'")   # prime → apostrophe
        s = s.replace('\u2018', "'")                          # left single quote
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _split_segments(text: str) -> List[Tuple[str, str, str]]:
        """
        Stage 2: Split a potentially multi-ligand string into segments.

        Splitting rules:
          1. Split on ';' (multi-ligand entries)
          2. For each part, separate the IUPAC name from annotations:
             - Bracket annotation: "4,4'-tBu-bpy" from "2,2'-bipyridine [4,4'-tBu-bpy]"
             - Parenthetical: "bipy" from "2,2'-bipyridine (bipy)"
             - Trailing notes after "for", "used as", "alongside" are stripped
          3. For L-codes (L1, L3, etc.), the parenthetical hint is preserved
             separately as it contains the scaffold keyword.

        Returns list of (cleaned_name, bracket_abbrev, lcode_hint) tuples.
        """
        raw_parts = re.split(r'\s*;\s*', text)

        # Precompile: keywords that signal a parenthetical is a description
        # (not part of the chemical name), and scaffold keywords that MUST
        # be preserved even inside parentheses.
        _DESC_STARTERS = re.compile(
            r'^(?:an |specific |chiral |newly |spiro-)', re.I
        )
        _SCAFFOLD_KW = re.compile(
            r'(?:oxazol|imidazol|phenanthrol|bipyridin|pyridin|pyridyl|'
            r'picolin|pyrazol|carboxamidin|amidin|amidoxim|terpyridin|'
            r'bis\(oxa|bis\(imi|oxazole|imidazole)',
            re.I,
        )

        segments = []
        for part in raw_parts:
            part = part.strip()
            if not part:
                continue

            # Extract bracket abbreviation
            # Scan ALL [...] brackets, pick the LAST one that looks like
            # a ligand label. Position-only brackets like [1,2-d] contain
            # only digits, commas, hyphens, spaces, and at most one
            # lowercase letter suffix. Real labels have uppercase or
            # multiple alphabetic characters.
            bracket_abbrev = ''
            for bm in re.finditer(r'\[([^\]]+)\]', part):
                candidate = bm.group(1).strip()
                has_upper = bool(re.search(r'[A-Z]', candidate))
                is_positional = bool(re.match(r'^[\d,\-\s]+[a-z]?$', candidate))
                if not is_positional:
                    bracket_abbrev = candidate

            # Detect L-code with parenthetical hint BEFORE stripping parens
            # e.g. "L3 (box/bis(oxazoline) ligand)" or "L1 (pyridine-imidazoline)"
            lcode_hint = ''
            lcode_m = re.match(r'^([A-Za-z]\d\w*)\s*\(([^)]+)\)', part, re.I)
            if lcode_m:
                lcode_hint = lcode_m.group(2).strip()

            # Remove bracket labels (keep chemical notation like [1,2-d]oxazole)
            def _is_label(m):
                c = m.group(1)
                return not bool(re.match(r'^[\d,\-\s]+[a-z]?$', c))
            cleaned = re.sub(r'\[([^\]]+)\]', lambda m: '' if _is_label(m) else m.group(0), part)

            # Chemical-structure-aware paren removal:
            # Only remove parens that look like ABBREVIATIONS or DESCRIPTIONS.
            # PRESERVE parens that contain scaffold keywords — these are part
            # of the chemical name (e.g. "bis(4-phenyl-4,5-dihydrooxazole)").
            def _should_strip_paren(m):
                content = m.group(1).strip()
                # Preserve if it contains any scaffold keyword
                if _SCAFFOLD_KW.search(content):
                    return False  # keep — structural parens
                # Preserve if it's long and contains chemical-looking text
                # (has numbers + letters mixed, typical of IUPAC substituents)
                if len(content) > 5 and re.search(r'[a-z].*\d|\d.*[a-z]', content):
                    return False
                # Strip: short abbreviations like "(bipy)", "(phen)", "(bpp)"
                # and description hints like "(an N,N-ligand)"
                return True

            cleaned = re.sub(r'\s*\(([^)]+)\)', lambda m: '' if _should_strip_paren(m) else m.group(0), cleaned)

            # Strip trailing usage notes (these commonly appear in reactions)
            cleaned = re.split(
                r'\s+(?:for|used as|alongside|as|with|optimal for|identified as)\b',
                cleaned, flags=re.I
            )[0].strip()
            # Also strip "complex" and "precatalyst" tails
            cleaned = re.split(
                r'\s+(?:complex|precatalyst|pre-catalyst|preformed)\b',
                cleaned, flags=re.I
            )[0].strip()

            if cleaned:
                segments.append((cleaned, bracket_abbrev, lcode_hint))

        return segments

    # ------------------------------------------------------------------
    # Bracket abbreviation lookup
    # ------------------------------------------------------------------

    # Pattern to detect canonical class names embedded in descriptions
    _EMBEDDED_CLASS_RE = re.compile(
        r'\b(' + '|'.join(re.escape(c) for c in VALID_CLASSES) + r')\b', re.I
    )

    def _classify_by_bracket(self, abbrev: str) -> Optional[ClassMatch]:
        """Stage 3: Match a bracket abbreviation against the data-driven + hardcoded index.

        Enhanced with three additional resolution strategies:
        1. Embedded class names in descriptions ("specific BiIM ligand")
        2. Compound abbreviation decomposition ("Pent(3,3)-Bis(IndOx)")
        3. Sub-token matching after splitting on hyphens/commas
        """
        if not abbrev:
            return None

        # --- 3a. Direct exact match (case-insensitive, prime-normalized) ---
        key = abbrev.lower().strip()
        for candidate_key in (key, self._prime_normalize(key)):
            if candidate_key in self._abbrev_index:
                cls = self._abbrev_index[candidate_key]
                if cls in VALID_CLASSES:
                    return ClassMatch(
                        cls=cls,
                        method=ClassificationMethod.BRACKET_ABBREV,
                        matched_term=abbrev,
                        confidence=5,
                    )

        # --- 3b. Embedded class name in description text ---
        #     e.g. "specific BiIM ligand", "specific BOX ligand"
        cn_match = self._EMBEDDED_CLASS_RE.search(abbrev)
        if cn_match:
            matched_name = cn_match.group(1)
            for vc in VALID_CLASSES:
                if vc.lower() == matched_name.lower():
                    return ClassMatch(
                        cls=vc,
                        method=ClassificationMethod.BRACKET_ABBREV,
                        matched_term=matched_name,
                        confidence=4,
                    )

        # --- 3c. Compound abbreviation decomposition ---
        #     e.g. "Pent(3,3)-Bis(IndOx)" → try "Pent", "IndOx", "Bis"
        #     Remove parenthetical position specifiers first
        cleaned = re.sub(r'\([^)]*\)', '', abbrev)
        parts = re.split(r'[-,/\s]+', cleaned)
        for part in parts:
            part = part.strip()
            if len(part) < 2:
                continue
            pk = part.lower()
            if pk in self._abbrev_index and self._abbrev_index[pk] in VALID_CLASSES:
                return ClassMatch(
                    cls=self._abbrev_index[pk],
                    method=ClassificationMethod.BRACKET_ABBREV,
                    matched_term=part,
                    confidence=4,
                )

        # --- 3d. Scaffold rules on the full bracket text ---
        m = self._classify_by_scaffold(key)
        if m and m.cls in VALID_CLASSES:
            return ClassMatch(
                cls=m.cls,
                method=ClassificationMethod.BRACKET_ABBREV,
                matched_term=abbrev,
                confidence=3,
            )

        return None

    # ------------------------------------------------------------------
    # Structural scaffold detection
    # ------------------------------------------------------------------

    def _classify_by_scaffold(self, text: str) -> Optional[ClassMatch]:
        """Stage 4: Run scaffold rules in priority order, first match wins."""
        for pattern, cls, confidence, desc in self._SCAFFOLD_RULES:
            m = pattern.search(text)
            if m:
                return ClassMatch(
                    cls=cls,
                    method=ClassificationMethod.SCAFFOLD_RULE,
                    matched_term=m.group(0),
                    confidence=confidence,
                )
        return None

    # ------------------------------------------------------------------
    # Abbreviation pattern matching
    # ------------------------------------------------------------------

    def _classify_by_abbrev_pattern(self, text: str) -> Optional[ClassMatch]:
        """Stage 5: Match inline abbreviation patterns."""
        best: Optional[ClassMatch] = None
        for pattern, cls, confidence in self._ABBREV_PATTERNS:
            m = pattern.search(text)
            if m:
                if best is None or confidence > best.confidence:
                    best = ClassMatch(
                        cls=cls,
                        method=ClassificationMethod.ABBREV_PATTERN,
                        matched_term=m.group(0),
                        confidence=confidence,
                    )
        return best

    # ------------------------------------------------------------------
    # Canonical class name detection
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_by_canonical(text: str) -> Optional[ClassMatch]:
        """Check if any canonical class code appears as a word in text."""
        for cls in VALID_CLASSES:
            # Use word-boundary matching for short codes (Bpy, BiOX etc.)
            if re.search(r'\b' + re.escape(cls) + r'\b', text, re.I):
                return ClassMatch(
                    cls=cls,
                    method=ClassificationMethod.CANONICAL_EXACT,
                    matched_term=cls,
                    confidence=5,
                )
        return None

    # ------------------------------------------------------------------
    # Public API: classify a ligand name
    # ------------------------------------------------------------------

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify a ligand name string into one or more database classes.

        Handles multi-ligand entries (e.g. "bipy for X; phen for Y") by
        splitting into segments and classifying each independently.

        Returns a ClassificationResult with primary_class, all_classes,
        matches, and parsed_refs (structured parse of the Optimum Ligand format).
        """
        normalized = self._normalize(text)

        # Stage 0: Parse structured Optimum Ligand format
        parsed_refs = self.parse_optimum_ligand(normalized)

        # Stage 0.5: ReactionLigandIndexer fast-path
        # If this exact text was pre-resolved at load time, use it directly.
        # This handles the common case where classify() is called on
        # Optimum Ligand strings that were already processed during load().
        if self.reaction_indexer:
            for ref in parsed_refs:
                cached = self.reaction_indexer.lookup_raw(ref.raw_segment)
                if cached and cached in VALID_CLASSES:
                    return ClassificationResult(
                        primary_class=cached,
                        all_classes=[cached],
                        matches=[ClassMatch(
                            cls=cached,
                            method=ClassificationMethod.LIGAND_DB_MATCH,
                            matched_term=ref.raw_segment[:50],
                            confidence=5,
                        )],
                        cleaned_name=ref.full_name or normalized[:80],
                        parsed_refs=parsed_refs,
                    )

        segments = self._split_segments(normalized)

        all_matches: List[ClassMatch] = []
        primary_name = ''

        for seg_idx, (segment_text, bracket_abbrev, lcode_hint) in enumerate(segments):
            if seg_idx == 0:
                primary_name = segment_text

            # Stage 3: Bracket abbreviation (highest priority)
            match = self._classify_by_bracket(bracket_abbrev)

            # Stage 3.5: Canonical class code in text
            if match is None:
                match = self._classify_by_canonical(segment_text)

            # Stage 4: Structural scaffold (now with preserved chemical parens)
            if match is None:
                match = self._classify_by_scaffold(segment_text)

            # Stage 4.5: L-code hint — classify the preserved parenthetical text
            # e.g. "L3 (box/bis(oxazoline) ligand)" → hint = "box/bis(oxazoline) ligand"
            if match is None and lcode_hint:
                match = self._classify_by_scaffold(lcode_hint)
                if match is None:
                    match = self._classify_by_abbrev_pattern(lcode_hint)
                # If the hint has ';' or '/', try each sub-part
                if (match is None or match.cls not in VALID_CLASSES) and lcode_hint:
                    for sub in re.split(r'[/;,]', lcode_hint):
                        sub = sub.strip()
                        if not sub or len(sub) < 3:
                            continue
                        m = self._classify_by_scaffold(sub)
                        if m and m.cls in VALID_CLASSES:
                            match = m
                            break
                    if match is None or match.cls not in VALID_CLASSES:
                        m2 = self._classify_by_abbrev_pattern(lcode_hint)
                        if m2 and m2.cls in VALID_CLASSES:
                            match = m2


            # Stage 5: Abbreviation pattern
            if match is None:
                match = self._classify_by_abbrev_pattern(segment_text)

            if match:
                all_matches.append(match)

        # Stage 6: Resolution — merge results
        if not all_matches:
            return ClassificationResult(
                primary_class='unknown',
                all_classes=[],
                matches=[],
                cleaned_name=primary_name,
            )

        # Deduplicate preserving order
        seen: Set[str] = set()
        unique_classes: List[str] = []
        for m in all_matches:
            if m.cls not in seen and m.cls in VALID_CLASSES:
                seen.add(m.cls)
                unique_classes.append(m.cls)

        # Primary = highest confidence; tie-break by first occurrence
        best = max(all_matches, key=lambda m: (m.confidence, -all_matches.index(m)))

        return ClassificationResult(
            primary_class=best.cls if best.cls in VALID_CLASSES else 'unknown',
            all_classes=unique_classes,
            matches=all_matches,
            cleaned_name=primary_name,
            parsed_refs=parsed_refs,
        )

    # ------------------------------------------------------------------
    # Data-driven abbreviation index
    # ------------------------------------------------------------------

    def build_abbrev_index(self, ligand_records: List[Dict[str, str]]):
        """Extend the abbreviation → class index from ligand_database.

        Called after ligands are loaded.  Supplements the hardcoded
        ABBREV_TO_CLASS with entries extracted from the actual 238-ligand
        database abbreviations (e.g. '(R,R)-Ph-BiIM' → BiIM).

        Parameters
        ----------
        ligand_records : list of dict
            Each dict must have 'abbreviation' (str) and 'class' (str) keys.
        """
        for rec in ligand_records:
            abbr = rec.get('abbreviation', '').strip()
            cls = rec.get('class', '').strip()
            if not abbr or cls not in VALID_CLASSES:
                continue
            key = self._prime_normalize(abbr.lower())
            # Only add if not already present (hardcoded entries take precedence)
            if key not in self._abbrev_index:
                self._abbrev_index[key] = cls

        logger.debug(
            "LigandClassifier: abbrev index now has %d entries (was %d hardcoded)",
            len(self._abbrev_index), len(self.ABBREV_TO_CLASS),
        )

    # ------------------------------------------------------------------
    # Stage 0: Optimum Ligand format parser
    # ------------------------------------------------------------------

    # Transition metal symbols — used to distinguish ligand abbreviation
    # brackets from coordination-complex formula brackets.
    _METAL_SYMBOLS_RE = re.compile(
        r'\b(Ni|Co|Fe|Cu|Pd|Pt|Zn|Mn|Cr|Rh|Ir|Ru|Os|Ag|Au|Al|Ti|V)\b'
    )

    def parse_optimum_ligand(self, raw: str) -> List[ParsedLigandRef]:
        """Parse the structured Optimum Ligand format.

        Canonical format::

            full_name [abbreviation] explanation

        - Multiple ligands separated by ``;`` (bracket-aware).
        - Dual-ligand systems also use ``+`` as separator.
        - Some entries use parenthetical abbreviations: ``name (abbr)``.
        - Paper-specific codes like ``L1 (description)`` are handled.

        Returns a list of :class:`ParsedLigandRef` objects, one per ligand.
        """
        if not raw or not raw.strip():
            return []

        normalized = self._normalize(raw)
        segments = self._split_on_separators(normalized)

        results: List[ParsedLigandRef] = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            ref = self._parse_single_ref(seg)
            if ref and (ref.full_name or ref.abbreviation):
                results.append(ref)
        return results

    def _split_on_separators(self, text: str) -> List[str]:
        """Split on ``;`` and ``+`` at the top level (not inside brackets/parens).

        This prevents incorrect splits inside IUPAC names like
        ``indeno[1,2-d]oxazole`` or complex formulas like
        ``[NiCl2(bpy)(H2O)2]``.
        """
        segments: List[str] = []
        current: List[str] = []
        depth_b = 0   # [ ]
        depth_p = 0   # ( )

        for ch in text:
            if ch == '[':
                depth_b += 1
            elif ch == ']':
                depth_b = max(0, depth_b - 1)
            elif ch == '(':
                depth_p += 1
            elif ch == ')':
                depth_p = max(0, depth_p - 1)

            if ch in (';', '+') and depth_b == 0 and depth_p == 0:
                segments.append(''.join(current))
                current = []
            else:
                current.append(ch)

        if current:
            segments.append(''.join(current))

        return segments

    def _find_abbreviation_bracket(self, text: str) -> Optional[re.Match]:
        """Find the ``[...]`` bracket most likely containing a ligand abbreviation.

        Heuristics (applied in order, first passing bracket wins):

        1. The character before ``[`` must be whitespace, comma, or semicolon.
           (IUPAC nomenclature brackets like ``indeno[1,2-d]`` have a letter before ``[``.)
        2. The content must NOT contain transition metal element symbols.
           (Coordination complex formulas like ``[NiCl2(bpy)Cl2]`` contain ``Ni``, ``Cl``.)
        3. If no bracket passes the filter, fall back to the first bracket.
        """
        matches = list(re.finditer(r'\[([^\]]+)\]', text))
        if not matches:
            return None

        for m in matches:
            start = m.start()
            # Rule 1: char before '[' must be whitespace/punctuation
            if start > 0 and text[start - 1] not in (' ', ',', ';', '\t', ':'):
                continue  # likely IUPAC nomenclature bracket
            # Rule 2: no transition metal symbols
            if self._METAL_SYMBOLS_RE.search(m.group(1)):
                continue  # likely a coordination complex formula
            return m

        # Fallback: return the first bracket
        return matches[0]

    def _find_abbreviation_paren(self, text: str) -> Optional[re.Match]:
        """Find a ``(...)`` that looks like a ligand abbreviation.

        Only considers parentheticals in the "name portion" of the text —
        i.e. before any usage keywords like "for", "used as", "as",
        "alongside", "with", "complex".

        Heuristics for qualifying parentheticals:
        - Content must be short (< 30 chars)
        - Must contain at least one letter
        - Must NOT look like a pure positional indicator (e.g. ``(1,2-d)``)
        - Must NOT look like a long description
        - Must NOT be embedded in a word (char before must be whitespace/punct)
        - Prefer the LAST qualifying parenthetical in the name portion
        """
        # Isolate the name portion (before usage keywords)
        name_portion = re.split(
            r'\s+(?:for|used as|alongside|as |with |complex|preformed|pre-formed)\b',
            text, flags=re.I
        )[0].strip()

        matches = list(re.finditer(r'\(([^)]+)\)', name_portion))
        if not matches:
            return None

        for m in reversed(matches):
            content = m.group(1).strip()
            # Skip if too long for an abbreviation
            if len(content) > 30:
                continue
            # Skip pure positional indicators: digits, commas, hyphens, optional single letter
            if re.match(r'^[\d,\-\s]+[a-z]?$', content):
                continue
            # Skip if char before '(' is a letter/digit (embedded in a word, e.g. NiCl2(bpy))
            start = m.start()
            if start > 0 and name_portion[start - 1].isalnum():
                continue
            # Must contain at least one letter
            if not re.search(r'[a-zA-Z]', content):
                continue
            return m

        return None

    @staticmethod
    def _is_non_ligand(text: str) -> bool:
        """Return True if the text represents a non-ligand entry."""
        text_lower = text.lower().strip()
        return any(
            pattern in text_lower
            for pattern in (
                'no exogenous ligand', 'ligand-free',
                'no ligand', 'no external ligand',
            )
        )

    def _parse_single_ref(self, seg: str) -> Optional[ParsedLigandRef]:
        """Parse a single ligand reference segment into structured components.

        Tries in order:
        1. ``[...]`` bracket as abbreviation
        2. ``(...)`` parenthetical as abbreviation (not embedded in words)
        3. L-code pattern: "L1 (description)" — description used for classification
        4. No structured abbreviation — treat entire segment as name
        """
        seg = seg.strip()
        if not seg or self._is_non_ligand(seg):
            return None

        # --- Try bracket abbreviation ---
        bracket_match = self._find_abbreviation_bracket(seg)
        if bracket_match:
            abbr = bracket_match.group(1).strip()
            full_name = seg[:bracket_match.start()].strip()
            # Remove trailing punctuation/comma from name
            full_name = full_name.rstrip(',;:').strip()
            explanation = seg[bracket_match.end():].strip().lstrip(',;:').strip()
            return ParsedLigandRef(
                full_name=full_name,
                abbreviation=abbr,
                explanation=explanation,
                raw_segment=seg,
            )

        # --- Check for L-code pattern: "L1 (description)" ---
        lcode_m = re.match(r'^([A-Za-z]\d\w*)\s*\(([^)]+)\)', seg, re.I)
        if lcode_m:
            code = lcode_m.group(1).strip()
            hint = lcode_m.group(2).strip()
            rest = seg[lcode_m.end():].strip().lstrip(',;:').strip()
            return ParsedLigandRef(
                full_name=code,
                abbreviation='',
                explanation=hint + ('; ' + rest if rest else ''),
                raw_segment=seg,
            )

        # --- Try parenthetical abbreviation ---
        paren_match = self._find_abbreviation_paren(seg)
        if paren_match:
            abbr = paren_match.group(1).strip()
            full_name = seg[:paren_match.start()].strip()
            full_name = full_name.rstrip(',;:').strip()
            explanation = seg[paren_match.end():].strip().lstrip(',;:').strip()
            return ParsedLigandRef(
                full_name=full_name,
                abbreviation=abbr,
                explanation=explanation,
                raw_segment=seg,
            )

        # --- No structured abbreviation ---
        # Try to separate name from trailing description on first top-level comma
        full_name, explanation = self._split_name_description(seg)
        return ParsedLigandRef(
            full_name=full_name,
            abbreviation='',
            explanation=explanation,
            raw_segment=seg,
        )

    @staticmethod
    def _split_name_description(text: str) -> Tuple[str, str]:
        """Separate a ligand name from trailing description.

        Splits on the first comma that is not inside parentheses or brackets.
        Only splits if the name portion is at least 5 characters (to avoid
        splitting short abbreviations).
        """
        text = text.strip()
        depth = 0
        for i, ch in enumerate(text):
            if ch in ('(', '['):
                depth += 1
            elif ch in (')', ']'):
                depth -= 1
            elif ch == ',' and depth == 0 and i > 5:
                return text[:i].strip(), text[i + 1:].strip()
        return text, ''

    # ------------------------------------------------------------------
    # Public API: detect classes in a user query
    # ------------------------------------------------------------------

    def detect_classes(self, query: str) -> List[str]:
        """
        Extract all ligand class names mentioned in a user query.

        Used by search_combined() to determine which classes to retrieve.
        Returns a deduplicated list in detection order.

        Enhanced: also parses structured Optimum Ligand format to extract
        abbreviation-based classes with higher priority.
        """
        normalized = self._normalize(query)

        found: List[str] = []
        seen: Set[str] = set()

        # --- Phase 1: Parse structured format (highest priority) ---
        # Extract bracket abbreviations from parsed ligand references
        parsed = self.parse_optimum_ligand(query)
        for ref in parsed:
            if ref.abbreviation:
                match = self._classify_by_bracket(ref.abbreviation)
                if match and match.cls in VALID_CLASSES and match.cls not in seen:
                    found.append(match.cls)
                    seen.add(match.cls)

        # --- Phase 2: Check canonical class codes ---
        for cls in VALID_CLASSES:
            if re.search(r'\b' + re.escape(cls) + r'\b', normalized, re.I):
                if cls not in seen:
                    found.append(cls)
                    seen.add(cls)

        # --- Phase 3: Run scaffold rules on the full query ---
        for pattern, cls, confidence, _ in self._SCAFFOLD_RULES:
            if pattern.search(normalized) and cls not in seen and cls in VALID_CLASSES:
                found.append(cls)
                seen.add(cls)

        # --- Phase 4: Run abbrev patterns on the full query ---
        for pattern, cls, confidence in self._ABBREV_PATTERNS:
            if pattern.search(normalized) and cls not in seen and cls in VALID_CLASSES:
                found.append(cls)
                seen.add(cls)

        # --- Phase 5: Scan ALL bracket abbreviations in the full query ---
        for bracket_m in re.finditer(r'\[([^\]]+)\]', normalized):
            abbrev = bracket_m.group(1).strip().lower()
            if abbrev in self._abbrev_index:
                cls = self._abbrev_index[abbrev]
                if cls not in seen and cls in VALID_CLASSES:
                    found.append(cls)
                    seen.add(cls)

        return found

    # ------------------------------------------------------------------
    # Public API: resolve a query to a database ligand name
    # ------------------------------------------------------------------

    @staticmethod
    def _prime_normalize(s: str) -> str:
        """Normalize all prime/apostrophe variants to a single form."""
        return re.sub(r"[\u2032'\u2019\u2018]", '\'', s)

    @staticmethod
    def resolve_ligand_name(
        query: str,
        db_names: List[str],
    ) -> Optional[str]:
        """
        Map a user query to the best-matching ligand name in the database.

        Resolution strategy (in priority order):
          1. Exact match (case-insensitive, prime-normalized)
          2. Bracket abbreviation match (e.g. [dtbbpy] → find DB name containing it)
          3. Normalized token overlap (Jaccard-like) — handles IUPAC variations
          4. Substring containment (longest common substring)

        Returns the database ligand name (lowercase) or None.
        """
        q = LigandClassifier._normalize(query).lower()
        q_norm = LigandClassifier._prime_normalize(q)

        # Build prime-normalized lookup: db_name_normalized → db_name_original
        db_norm_map = {}
        for dbn in db_names:
            db_norm_map[LigandClassifier._prime_normalize(dbn)] = dbn
        db_norm_set = set(db_norm_map.keys())

        # 1. Exact match (prime-normalized)
        if q_norm in db_norm_set:
            return db_norm_map[q_norm]

        # 2. Bracket abbreviation → find in DB names
        bracket_match = re.search(r'\[([^\]]+)\]', query)
        if bracket_match:
            abbrev = bracket_match.group(1).strip().lower()
            abbrev_norm = LigandClassifier._prime_normalize(abbrev)
            for db_norm, db_orig in db_norm_map.items():
                if abbrev_norm in db_norm:
                    return db_orig

        # 3. Token overlap scoring
        def _tokens(s: str) -> Set[str]:
            """Extract meaningful chemical tokens from a name."""
            s = LigandClassifier._prime_normalize(s)
            # Split on spaces, hyphens, commas, parentheses, brackets
            raw = re.split(r'[\s,\-()\[\]{}]+', s)
            # Filter: keep tokens that are chemical (letters, maybe with numbers/subscripts)
            tokens = set()
            for t in raw:
                t = t.strip().lower()
                if len(t) >= 2 and re.search(r'[a-z]', t):
                    tokens.add(t)
            return tokens

        q_tokens = _tokens(q)
        if q_tokens:
            best_name = None
            best_score = 0.0
            for dbn in db_names:
                db_tokens = _tokens(dbn)
                if not db_tokens:
                    continue
                intersection = q_tokens & db_tokens
                union = q_tokens | db_tokens
                jaccard = len(intersection) / len(union) if union else 0
                # Bonus for longer intersection (more specific match)
                score = jaccard + 0.1 * len(intersection)
                if score > best_score:
                    best_score = score
                    best_name = dbn
            if best_name and best_score >= 0.15:
                return best_name

        # 4. Longest common substring containment
        best_name = None
        best_overlap = 0
        q_clean = LigandClassifier._prime_normalize(re.sub(r"[()]", '', q).lower())
        for dbn in db_names:
            dbn_clean = LigandClassifier._prime_normalize(re.sub(r"[()]", '', dbn).lower())
            # Check both directions
            if q_clean in dbn_clean or dbn_clean in q_clean:
                overlap = min(len(q_clean), len(dbn_clean))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_name = dbn
        if best_name:
            return best_name

        return None


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LigandProperty:
    """DFT-computed electronic descriptors for a single ligand."""
    name: str
    ligand_class: str
    abbreviation: str = ""
    homo: float = 0.0
    lumo: float = 0.0
    gap: float = 0.0
    omega: float = 0.0       # electrophilicity index
    i_min: float = 0.0       # ionization potential
    v_min: float = 0.0       # electron affinity
    r1_homa: float = 0.0     # aromaticity index, ring 1
    r2_homa: float = 0.0     # aromaticity index, ring 2


@dataclass
class ReactionLigandEntry:
    """A literature entry linking a reaction to its optimum ligand."""
    doi: str
    title: str = ""
    optimum_ligand: str = ""
    coupling_partner: str = ""
    ligand_knowledge: str = ""
    mapped_class: str = ""  # resolved ligand class name


# ═══════════════════════════════════════════════════════════════════════════
# Database service
# ═══════════════════════════════════════════════════════════════════════════

class RedCrossDatabase:
    """
    Database service for the RedCross (Reductive Coupling) subsystem.

    Loads two JSON sources, builds text search indices, and provides
    retrieval methods for both ligand properties and reaction-ligand
    literature data.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'redcross_data'
            )
        self.data_dir = Path(data_dir)
        self.classifier = LigandClassifier()

        # Ligand electronic properties (from ligand_database.json)
        self.ligands: List[LigandProperty] = []
        self._ligand_by_name: Dict[str, LigandProperty] = {}
        self._abbr_to_name: Dict[str, str] = {}      # abbreviation → canonical name
        self._ligands_by_class: Dict[str, List[LigandProperty]] = {}

        # Reaction-ligand literature (from reaction_database.json)
        self.reactions: List[ReactionLigandEntry] = []
        self._reactions_by_doi: Dict[str, ReactionLigandEntry] = {}

        # Search indices
        self._name_index: Dict[str, List[int]] = {}       # ligand name -> indices
        self._class_index: Dict[str, List[int]] = {}      # class -> indices
        self._doi_index: Dict[str, int] = {}              # doi -> reaction index
        self._keyword_index: Dict[str, List[int]] = {}    # word -> reaction indices
        self._ligand_kw_index: Dict[str, List[int]] = {}  # word -> ligand indices

        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load all data from JSON files and build search indices.

        Uses stdlib json module only — no pandas / C extensions.

        Loading sequence (each step feeds into the next):
          1. Load ligand_database.json → populate self.ligands
          2. Build Layer 2 abbrev index from ligand DB abbreviations
          3. Load reaction_database.json → populate self.reactions
          4. Build ReactionLigandIndexer (Layer 3) from reaction entries,
             cross-referencing against the ligand DB.  This also extends
             the classifier's abbrev index with any newly discovered
             abbreviation → class mappings.
          5. Attach the indexer to the classifier for fast-path lookups.
          6. Build keyword search indices.
        """
        if self._loaded:
            return True

        try:
            ligand_path = self.data_dir / 'ligand_database.json'
            reaction_path = self.data_dir / 'reaction_database.json'

            if ligand_path.exists():
                self._load_ligands_json(ligand_path)
            else:
                logger.error("ligand_database.json not found in %s", self.data_dir)

            # Layer 2: Build data-driven abbreviation index from loaded ligands
            if self.ligands:
                self.classifier.build_abbrev_index([
                    {'abbreviation': lp.abbreviation, 'class': lp.ligand_class}
                    for lp in self.ligands
                ])

            if reaction_path.exists():
                self._load_reactions_json(reaction_path)
            else:
                logger.error("reaction_database.json not found in %s", self.data_dir)

            if not self.ligands and not self.reactions:
                logger.error("RedCross database: no data loaded")
                return False

            # Layer 3: Build reaction-derived index
            # Prepare lookup maps from the ligand database
            ligand_name_to_class = {
                LigandClassifier._prime_normalize(lp.name.lower()): lp.ligand_class
                for lp in self.ligands
                if lp.name and lp.ligand_class in VALID_CLASSES
            }
            ligand_abbr_to_class = {
                LigandClassifier._prime_normalize(lp.abbreviation.lower()): lp.ligand_class
                for lp in self.ligands
                if lp.abbreviation and lp.ligand_class in VALID_CLASSES
            }

            # Temporarily load raw reaction data for the indexer
            if reaction_path.exists():
                with open(reaction_path, 'r', encoding='utf-8') as f:
                    raw_reactions = json.load(f)

                indexer = ReactionLigandIndexer()
                indexer.build(
                    reaction_records=raw_reactions,
                    ligand_name_to_class=ligand_name_to_class,
                    ligand_abbr_to_class=ligand_abbr_to_class,
                    classifier=self.classifier,
                )
                self.classifier.reaction_indexer = indexer
                self._reaction_indexer = indexer

                logger.info(
                    "ReactionLigandIndexer: %s", indexer.get_stats()
                )

            self._build_indices()
            self._loaded = True
            logger.info(
                "RedCross Database loaded (JSON): "
                "%d ligands, %d reactions (%d curated), "
                "abbrev index: %d entries",
                len(self.ligands), len(self.reactions),
                sum(1 for r in self.reactions if r.title),
                len(self.classifier._abbrev_index),
            )
            return True

        except (OSError, ValueError, KeyError, TypeError) as e:
            logger.error("Failed to load RedCross database: %s", e)
            return False

    # ------------------------------------------------------------------
    # JSON loaders (stdlib-only — NO pandas, NO C extensions)
    # ------------------------------------------------------------------

    def _load_ligands_json(self, path: Path):
        """Load ligand electronic properties from ligand_database.json.

        Expected JSON schema (one object per row, produced by
        ``pandas.to_json(orient='records')`` from ligand_database.xlsx):
            Ligand Name          — str, full IUPAC / descriptive name
            Ligand Abbreviation  — str, short code (not loaded into dataclass)
            Class                — str, one of Phen/Bpy/PyrOx/PyrIm/PyCam/BiOX/BiIM
            HOMO (eV)            — float
            LUMO (eV)            — float
            Gap (eV)             — float
            ω (eV)               — float (omega / electrophilicity index)
            I_min (eV)           — float (ionization potential)
            V_min (eV)           — float (electron affinity)
            R1-HOMA              — float (aromaticity, ring 1)
            R2-HOMA              — float (aromaticity, ring 2)
        """
        with open(path, 'r', encoding='utf-8') as f:
            rows = json.load(f)

        for row in rows:
            name = str(row.get('Ligand Name') or '').strip()
            if not name:
                continue

            cls = str(row.get('Class') or '').strip()

            abbr = str(row.get('Ligand Abbreviation') or '').strip()

            lp = LigandProperty(
                name=name,
                ligand_class=cls,
                abbreviation=abbr,
                homo=float(row.get('HOMO (eV)', 0)),
                lumo=float(row.get('LUMO (eV)', 0)),
                gap=float(row.get('Gap (eV)', 0)),
                omega=float(row.get('\u03c9 (eV)', 0)),
                i_min=float(row.get('I_min (eV)', 0)),
                v_min=float(row.get('V_min (eV)', 0)),
                r1_homa=float(row.get('R1-HOMA', 0)),
                r2_homa=float(row.get('R2-HOMA', 0)),
            )
            self.ligands.append(lp)
            self._ligand_by_name[lp.name.lower()] = lp
            self._ligands_by_class.setdefault(lp.ligand_class, []).append(lp)
            if abbr:
                self._abbr_to_name[abbr.lower()] = lp.name

    def _load_reactions_json(self, path: Path):
        """Load reaction-ligand literature from reaction_database.json.

        Expected JSON schema (one object per row, produced by
        ``pandas.to_json(orient='records')`` from reaction_database.xlsx):
            DOI              — str
            Title            — str
            Optimum Ligand   — str
            Reaction         — str (coupling partner / reaction description)
            Ligand Knowledge — str | null  (null for ~95 of 225 rows)
        """
        with open(path, 'r', encoding='utf-8') as f:
            rows = json.load(f)

        for row in rows:
            doi = str(row.get('DOI') or '').strip()
            if not doi or doi.lower() in ('nan', 'none', ''):
                continue

            title = str(row.get('Title') or '').strip()
            opt_lig = str(row.get('Optimum Ligand') or '').strip()
            partner = str(row.get('Reaction') or '').strip()
            # Ligand Knowledge can be None — convert to empty string
            raw_knowledge = row.get('Ligand Knowledge')
            knowledge = str(raw_knowledge).strip() if raw_knowledge is not None else ''

            # Use the classifier pipeline instead of CLASS_ALIASES
            result = self.classifier.classify(opt_lig)
            mapped_class = result.primary_class

            entry = ReactionLigandEntry(
                doi=doi, title=title, optimum_ligand=opt_lig,
                coupling_partner=partner, ligand_knowledge=knowledge,
                mapped_class=mapped_class,
            )
            self.reactions.append(entry)
            self._reactions_by_doi[doi] = entry

    # ------------------------------------------------------------------
    # Ligand class resolution (delegated to classifier)
    # ------------------------------------------------------------------

    def _resolve_ligand_class(self, ligand_name: str) -> str:
        """Map a reaction's optimum ligand name to the closest database class."""
        if not ligand_name:
            return ''
        return self.classifier.classify(ligand_name).primary_class

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_indices(self):
        """Build keyword search indices over both datasets."""
        # Ligand name index (word-level)
        for i, lp in enumerate(self.ligands):
            name_lower = lp.name.lower()
            self._name_index.setdefault(name_lower, []).append(i)
            # Also index individual meaningful tokens
            for token in re.findall(r'[a-z]{2,}', name_lower):
                if len(token) >= 2:
                    self._ligand_kw_index.setdefault(token, []).append(i)

        # Ligand class index
        for i, lp in enumerate(self.ligands):
            self._class_index.setdefault(lp.ligand_class, []).append(i)

        # Reaction indices
        for i, entry in enumerate(self.reactions):
            if entry.doi:
                self._doi_index[entry.doi] = i
            # Index keywords from title, ligand name, coupling partner
            text_fields = [entry.title, entry.optimum_ligand,
                           entry.coupling_partner, entry.ligand_knowledge]
            for field in text_fields:
                for word in re.findall(r'\w+', field.lower()):
                    if len(word) > 2:
                        self._keyword_index.setdefault(word, []).append(i)

    # ------------------------------------------------------------------
    # Search methods — Ligand properties
    # ------------------------------------------------------------------

    def search_ligands(self, query: str, limit: int = 10,
                       ligand_class: str = None) -> List[Dict[str, Any]]:
        """Search ligands by name, class, or arbitrary keyword."""
        query_lower = query.lower()
        results: List[Tuple[int, int]] = []  # (index, score)
        seen: set = set()

        def _add(idx, score):
            if idx not in seen:
                seen.add(idx)
                results.append((idx, score))

        # 0. Classifier-based name resolution (replaces LIGAND_NAME_ALIASES)
        #    Uses the pipeline: parse → bracket abbrev → canonical → scaffold → abbrev pattern
        db_names = list(self._name_index.keys())
        resolved = self.classifier.resolve_ligand_name(query, db_names)
        if resolved and resolved in self._name_index:
            for idx in self._name_index[resolved]:
                _add(idx, 120)

        # 0b. Try parsed abbreviation → ligand database abbreviation match
        parsed_refs = self.classifier.parse_optimum_ligand(query)
        for ref in parsed_refs:
            if ref.abbreviation:
                abbr_key = LigandClassifier._prime_normalize(ref.abbreviation.lower())
                if abbr_key in self._abbr_to_name:
                    canonical_name = self._abbr_to_name[abbr_key]
                    lp = self._ligand_by_name.get(canonical_name.lower())
                    if lp:
                        _add(self.ligands.index(lp), 125)

        # 1. Exact name match (highest score)
        for name, indices in self._name_index.items():
            if query_lower == name:
                for idx in indices:
                    _add(idx, 100)

        # 2. Substring name match
        for name, indices in self._name_index.items():
            if query_lower in name or name in query_lower:
                for idx in indices:
                    _add(idx, 50)

        # 3. Class name match
        for cls, indices in self._class_index.items():
            if re.search(r'\b' + re.escape(cls) + r'\b', query_lower, re.I):
                for idx in indices:
                    _add(idx, 40)

        # 4. Classifier-detected class match (replaces CLASS_ALIASES iteration)
        detected = self.classifier.detect_classes(query)
        for cls in detected:
            for idx in self._class_index.get(cls, []):
                _add(idx, 40)

        # 5. Keyword match on individual tokens
        for token in re.findall(r'[a-z]{2,}', query_lower):
            if token in self._ligand_kw_index:
                for idx in self._ligand_kw_index[token]:
                    _add(idx, 10)

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Filter by class if requested
        output = []
        for idx, score in results:
            lp = self.ligands[idx]
            if ligand_class and lp.ligand_class != ligand_class:
                continue
            output.append(self._ligand_to_dict(lp, score))
            if len(output) >= limit:
                break

        return output

    def get_ligand_classes(self) -> Dict[str, Any]:
        """Return all ligand classes with counts and descriptor ranges."""
        classes = {}
        for cls, ligands in self._ligands_by_class.items():
            homos = [l.homo for l in ligands]
            lumos = [l.lumo for l in ligands]
            gaps = [l.gap for l in ligands]
            omegas = [l.omega for l in ligands]
            classes[cls] = {
                'count': len(ligands),
                'ligand_names': [l.name for l in ligands],
                'HOMO_range': [min(homos), max(homos)],
                'LUMO_range': [min(lumos), max(lumos)],
                'Gap_range': [min(gaps), max(gaps)],
                'omega_range': [min(omegas), max(omegas)],
            }
        return classes

    def get_ligands_by_class(self, ligand_class: str) -> List[Dict[str, Any]]:
        """Return all ligands in a given class with full descriptors."""
        ligands = self._ligands_by_class.get(ligand_class, [])
        return [self._ligand_to_dict(lp) for lp in ligands]

    def get_ligand_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Look up a ligand by exact name (case-insensitive)."""
        lp = self._ligand_by_name.get(name.lower())
        if lp:
            return self._ligand_to_dict(lp)
        return None

    # Descriptor attribute names used for similarity computation
    _SIM_ATTRS = ('homo', 'lumo', 'gap', 'omega', 'i_min', 'v_min',
                  'r1_homa', 'r2_homa')

    def find_similar_ligands(self, query: str, top_k: int = 5,
                             same_class_only: bool = False
                             ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Find the ligand closest to *query* and return similar ligands.

        Returns a 2-tuple:
            (reference_dict, similar_list)

        *reference_dict* is the matched reference ligand (or None).
        *similar_list* is a list of dicts, each with the ligand data plus
        a ``similarity`` key (0–1, 1 = identical).

        The distance metric is normalised Euclidean distance over the 8
        electronic descriptors (HOMO, LUMO, Gap, omega, I_min, V_min,
        R1-HOMA, R2-HOMA).  Each descriptor is min-max normalised across
        the full 238-ligand dataset before computing distance.
        """
        # --- 1. Resolve the reference ligand via classifier pipeline ---
        ref_lp = None
        query_lower = query.lower()

        # Try exact name lookup first
        ref_lp_check = self._ligand_by_name.get(query_lower)
        if ref_lp_check:
            ref_lp = ref_lp_check

        # Try classifier-based name resolution
        if ref_lp is None:
            db_names = list(self._name_index.keys())
            resolved = self.classifier.resolve_ligand_name(query, db_names)
            if resolved:
                ref_lp = self._ligand_by_name.get(resolved)

        # Try substring match on ligand names (pick best = longest overlap)
        if ref_lp is None:
            best_name = None
            best_overlap = 0
            for name in self._name_index:
                if query_lower in name or name in query_lower:
                    overlap = min(len(query_lower), len(name))
                    if overlap > best_overlap or (
                        overlap == best_overlap and
                        (best_name is None or len(name) < len(best_name))
                    ):
                        best_name = name
                        best_overlap = overlap
            if best_name:
                ref_lp = self.ligands[self._name_index[best_name][0]]

        # Try class-level: pick the first ligand of the matched class
        if ref_lp is None:
            result = self.classifier.classify(query)
            cls = result.primary_class
            if cls in self._ligands_by_class:
                ref_lp = self._ligands_by_class[cls][0]

        if ref_lp is None:
            return None, []

        # --- 2. Precompute per-descriptor min/max for normalisation ---
        if not hasattr(self, '_sim_ranges'):
            ranges = {}
            for attr in self._SIM_ATTRS:
                vals = [getattr(lp, attr) for lp in self.ligands]
                lo, hi = min(vals), max(vals)
                ranges[attr] = (lo, hi) if hi != lo else (lo, lo + 1.0)
            self._sim_ranges = ranges

        # --- 3. Compute normalised Euclidean distance ---
        def _distance(a: LigandProperty, b: LigandProperty) -> float:
            sq_sum = 0.0
            for attr in self._SIM_ATTRS:
                lo, hi = self._sim_ranges[attr]
                na = (getattr(a, attr) - lo) / (hi - lo)
                nb = (getattr(b, attr) - lo) / (hi - lo)
                sq_sum += (na - nb) ** 2
            return math.sqrt(sq_sum)

        ref_vec = _distance(ref_lp, ref_lp)  # 0.0 — sanity check
        distances = []
        for lp in self.ligands:
            if lp is ref_lp:
                continue
            if same_class_only and lp.ligand_class != ref_lp.ligand_class:
                continue
            d = _distance(ref_lp, lp)
            distances.append((lp, d))

        distances.sort(key=lambda x: x[1])

        # Normalise by sqrt(n_attrs) — the theoretical max Euclidean
        # distance in the unit-normalised descriptor space.
        max_possible = math.sqrt(len(self._SIM_ATTRS))

        similar = []
        for lp, d in distances[:top_k]:
            sim_score = round(max(0.0, 1.0 - d / max_possible), 3)
            entry = self._ligand_to_dict(lp)
            entry['similarity'] = sim_score
            similar.append(entry)

        return self._ligand_to_dict(ref_lp), similar

    def compare_ligand_classes(self, class_a: str, class_b: str) -> Optional[Dict[str, Any]]:
        """Compare average electronic descriptors between two ligand classes."""
        a = self._ligands_by_class.get(class_a, [])
        b = self._ligands_by_class.get(class_b, [])
        if not a or not b:
            return None

        def _avg(lst, attr):
            vals = [getattr(x, attr) for x in lst]
            return round(sum(vals) / len(vals), 3)

        attrs = ['homo', 'lumo', 'gap', 'omega', 'i_min', 'v_min',
                 'r1_homa', 'r2_homa']
        comparison = {}
        for attr in attrs:
            comparison[attr] = {
                class_a: _avg(a, attr),
                class_b: _avg(b, attr),
                'diff': round(_avg(a, attr) - _avg(b, attr), 3),
            }
        comparison['count'] = {class_a: len(a), class_b: len(b)}
        return comparison

    # ------------------------------------------------------------------
    # Search methods — Reaction-Ligand literature
    # ------------------------------------------------------------------

    def search_reactions(self, query: str, limit: int = 10,
                         ligand_class: str = None) -> List[Dict[str, Any]]:
        """Search reaction-ligand entries by keyword (title, ligand, partner, knowledge)."""
        query_lower = query.lower()
        query_words = re.findall(r'\w+', query_lower)
        scores: Dict[int, int] = {}

        # Score by word overlap
        for word in query_words:
            if len(word) > 2 and word in self._keyword_index:
                for idx in self._keyword_index[word]:
                    scores[idx] = scores.get(idx, 0) + 1

        # Also do full-query substring on key fields
        for i, entry in enumerate(self.reactions):
            for field in [entry.title, entry.optimum_ligand,
                          entry.coupling_partner]:
                if query_lower in field.lower():
                    scores[i] = scores.get(i, 0) + 5

        # Sort by score
        sorted_indices = sorted(scores.items(), key=lambda x: x[1],
                                reverse=True)

        output = []
        for idx, score in sorted_indices:
            entry = self.reactions[idx]
            if not entry.title:
                continue
            if ligand_class and entry.mapped_class != ligand_class:
                continue
            output.append(self._reaction_to_dict(entry, score))
            if len(output) >= limit:
                break

        return output

    def get_reaction_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Look up a reaction entry by DOI."""
        entry = self._reactions_by_doi.get(doi)
        if entry:
            return self._reaction_to_dict(entry)
        return None

    def get_reactions_by_class(self, ligand_class: str) -> List[Dict[str, Any]]:
        """Get all curated reactions whose optimum ligand maps to a given class."""
        results = []
        for entry in self.reactions:
            if entry.mapped_class == ligand_class and entry.title:
                results.append(self._reaction_to_dict(entry))
        return results

    # ------------------------------------------------------------------
    # Cross-database search (the key RAG enabler)
    # ------------------------------------------------------------------

    def search_combined(self, query: str, max_ligands: int = 5,
                        max_reactions: int = 5) -> Dict[str, Any]:
        """
        Search both datasets simultaneously and return structured context.

        When ligand classes are detected in the query, this method:
        1. Detects ALL mentioned ligand classes (not just the first).
        2. Pulls the top ligands per detected class.
        3. Pulls matching reactions per detected class.
        4. Returns merged results for RAG prompt injection.
        """
        query_lower = query.lower()

        # Detect ALL mentioned class names via classifier pipeline
        detected_classes: List[str] = self.classifier.detect_classes(query)

        # Per-class limit scales with the number of detected classes
        per_class_ligands = max(2, max_ligands // max(len(detected_classes), 1))
        per_class_reactions = max(2, max_reactions // max(len(detected_classes), 1))

        all_ligands: List[Dict[str, Any]] = []
        all_reactions: List[Dict[str, Any]] = []
        seen_ligand_names: set = set()
        seen_dois: set = set()

        if detected_classes:
            for cls in detected_classes:
                # Ligands for this class
                cls_ligands = self.search_ligands(
                    query, limit=per_class_ligands, ligand_class=cls
                )
                for l in cls_ligands:
                    if l['name'] not in seen_ligand_names:
                        all_ligands.append(l)
                        seen_ligand_names.add(l['name'])

                # Reactions for this class
                cls_reactions = self.search_reactions(
                    query, limit=per_class_reactions, ligand_class=cls
                )
                for r in cls_reactions:
                    if r['doi'] not in seen_dois:
                        all_reactions.append(r)
                        seen_dois.add(r['doi'])

                # Supplement with class-level data if keyword search was thin
                if len(cls_ligands) < 2:
                    for l in self.get_ligands_by_class(cls):
                        if l['name'] not in seen_ligand_names:
                            all_ligands.append(l)
                            seen_ligand_names.add(l['name'])
                        if len(all_ligands) >= max_ligands:
                            break

                if len(cls_reactions) < 2:
                    for r in self.get_reactions_by_class(cls):
                        if r['doi'] not in seen_dois:
                            all_reactions.append(r)
                            seen_dois.add(r['doi'])
                        if len(all_reactions) >= max_reactions:
                            break
        else:
            # No class detected — fall back to unfiltered keyword search
            all_ligands = self.search_ligands(query, limit=max_ligands)
            all_reactions = self.search_reactions(query, limit=max_reactions)

        # For multi-class queries, compute class comparison data
        class_comparison = None
        if len(detected_classes) == 2:
            class_comparison = self.compare_ligand_classes(
                detected_classes[0], detected_classes[1]
            )

        return {
            'ligands': all_ligands,
            'reactions': all_reactions,
            'detected_class': detected_classes[0] if detected_classes else None,
            'detected_classes': detected_classes,
            'ligand_count': len(all_ligands),
            'reaction_count': len(all_reactions),
            'class_comparison': class_comparison,
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics for both datasets."""
        curated = sum(1 for r in self.reactions if r.title)
        return {
            'ligands': len(self.ligands),
            'ligand_classes': len(self._ligands_by_class),
            'class_names': list(self._ligands_by_class.keys()),
            'reactions_total': len(self.reactions),
            'reactions_curated': curated,
            'reactions_doi_only': len(self.reactions) - curated,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ligand_to_dict(lp: LigandProperty, score: int = 0) -> Dict[str, Any]:
        return {
            'name': lp.name,
            'abbreviation': lp.abbreviation,
            'class': lp.ligand_class,
            'HOMO_eV': lp.homo,
            'LUMO_eV': lp.lumo,
            'Gap_eV': lp.gap,
            'omega_eV': lp.omega,
            'I_min_eV': lp.i_min,
            'V_min_eV': lp.v_min,
            'R1_HOMA': lp.r1_homa,
            'R2_HOMA': lp.r2_homa,
            'relevance_score': score,
        }

    @staticmethod
    def _reaction_to_dict(entry: ReactionLigandEntry,
                          score: int = 0) -> Dict[str, Any]:
        return {
            'doi': entry.doi,
            'title': entry.title,
            'optimum_ligand': entry.optimum_ligand,
            'coupling_partner': entry.coupling_partner,
            'ligand_knowledge': entry.ligand_knowledge,
            'mapped_class': entry.mapped_class,
            'relevance_score': score,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_redcross_database: Optional[RedCrossDatabase] = None


def get_redcross_database() -> RedCrossDatabase:
    """Get or create the global RedCross database instance."""
    global _redcross_database
    if _redcross_database is None:
        _redcross_database = RedCrossDatabase()
        _redcross_database.load()
    return _redcross_database