"""
ReactionLens constants and prompts.
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RL_MAX_RETRIES: int = 3
RL_RETRY_DELAY: float = 3.0
RL_MAX_OUTPUT_TOKENS: int = 16384
RL_TEXT_CHUNK_SIZE: int = 8000
RL_MIN_PARAGRAPH_LENGTH: int = 80

# ---------------------------------------------------------------------------
# Reaction Detection Prompt
# ---------------------------------------------------------------------------

REACTION_DETECTION_PROMPT = """You are an expert chemist analyzing a paragraph from a scientific paper.

Your task is to determine whether this paragraph contains any chemical reaction information, and if so, extract it in full detail.

A paragraph CONTAINS a reaction if it describes ANY of the following:
- Reaction conditions (temperature, time, solvent, catalyst, ligand, atmosphere, pressure)
- Reactants, products, or intermediates with specific chemical identities
- Experimental procedures for carrying out a reaction
- Optimization or screening of reaction conditions
- Substrate scope or generality studies
- Yield, conversion, or selectivity data
- Reagent quantities, loadings, or concentrations

A paragraph does NOT contain a reaction if it only discusses:
- General introduction, background, or motivation
- Literature references or citations
- Spectral characterization without reaction context
- Biological assays or biological activity
- Purely theoretical or computational results without experimental conditions

If the paragraph DOES contain reaction information, extract ALL of it using the same structured format as ChemExtract:

{
  "has_reactions": true,
  "reactions": [
    {
      "id": "reaction_1",
      "type": "reaction type (e.g. cross-coupling, oxidation, reduction, C-H activation)",
      "entry": 1,
      "reactants": [{"name": "compound name", "smiles": "SMILES if determinable", "formula": "molecular formula if determinable"}],
      "products": [{"name": "compound name", "smiles": "SMILES if determinable", "formula": "molecular formula if determinable"}],
      "catalysts": [{"name": "catalyst name", "loading": "e.g. 5 mol%"}],
      "ligands": [{"name": "ligand name"}],
      "solvents": ["solvent name"],
      "reagents": ["reagent name with quantity"],
      "conditions": {
        "temperature": "e.g. 80 C",
        "time": "e.g. 12 h",
        "atmosphere": "N2/Ar/air",
        "pressure": "if applicable",
        "concentration": "if applicable"
      },
      "outcomes": {
        "yield": "e.g. 85%",
        "conversion": "if mentioned",
        "selectivity": {"ee": "...", "dr": "...", "regioselectivity": "..."}
      }
    }
  ],
  "compounds": [
    {
      "name": "compound name",
      "smiles": "SMILES if determinable",
      "formula": "molecular formula if determinable",
      "role": "reactant|product|catalyst|ligand|solvent|reagent|additive"
    }
  ],
  "scaffold_smiles": "SMILES of general scaffold with [*] placeholders if R-group table is mentioned",
  "rgroup_table": {
    "scaffold_label": "label (e.g. 1a)",
    "rgroups": {
      "R1": {"1a": "SMILES or description", "1b": "SMILES or description"},
      "Y":  {"1a": "Cl", "1b": "SPh"}
    },
    "partner_rgroups": {
      "R2": {"2a": "SMILES or description"}
    }
  },
  "experimental_procedures": ["procedural text if present"]
}

If the paragraph does NOT contain any reaction information, return:
{"has_reactions": false}

CRITICAL SMILES RULES:
- The "smiles" field MUST contain REAL, RDKit-parsable SMILES strings only.
- NEVER use placeholder notation like R0, R1, Ar, X, Y, Z in SMILES strings.
  WRONG: "R0-I", "RC(O)Cl", "ArCH2Cl"
  CORRECT: "CCCCI", "CC(=O)Cl", "c1ccc(CCl)cc1" (or null)
- If you cannot determine the exact SMILES from the text, set "smiles" to null.
- Common solvents always have real SMILES: THF = "C1CCOC2", DMF = "CN(C)C=O", etc.

IMPORTANT RULES:
- Extract EVERY individual reaction mentioned in the paragraph.
- Each distinct reaction condition set = one separate reaction entry.
- Include ALL conditions: temperature, time, solvent, atmosphere, catalyst loading.
- Include the entry/row number if mentioned in the text.
- Be EXHAUSTIVE — do not skip or summarize any reaction data."""
