import io
import base64
import logging
from typing import Optional

from flask import Blueprint, jsonify

from errors import APIError, NotFoundError

logger = logging.getLogger(__name__)

chemistry_bp = Blueprint('chemistry', __name__, url_prefix='/api')

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not installed. Molecular visualization features disabled.")

MOLECULES = {
    "phenyl_boronic": "B(c1ccccc1)(O)O",
    "1_bromopropane": "CCCBr",
    "tetramethyltin": "C[Sn](C)(C)C",
    "styrene": "C=Cc1ccccc1",
    "phenylacetylene": "C#Cc1ccccc1",
    "bromobenzene": "Brc1ccccc1",
    "phenylmagnesium_bromide": "Br[Mg]c1ccccc1",
    "phenylzinc_bromide": "[Zn]Brc1ccccc1",
    "tributylphenylstannane": "CCCC[Sn](CCCC)(CCCC)c1ccccc1",
    "iodobenzene": "Ic1ccccc1",
    "chlorobenzene": "Clc1ccccc1",
    "trimethylphenylsilane": "C[Si](C)(C)c1ccccc1",
    "morpholine": "C1COCCN1",
    "vinyl_triflate": "C=CC(=O)OS(=O)(=O)C(F)(F)F",
    "acrylate": "C=CC(=O)O",
    "tert_butyl_bromide": "CC(C)(C)Br",
    "ethyl_4_bromobenzoate": "CC(=O)Oc1ccc(Br)cc1",
    "4_bromoacetophenone": "Cc(=O)c1ccc(Br)cc1",
    "4_bromobenzaldehyde": "O=Cc1ccc(Br)cc1",
    "4_bromobenzonitrile": "N#Cc1ccc(Br)cc1",
    "anisole": "COc1ccccc1",
    "nitrobenzene": "O=[N+]([O-])c1ccccc1",
    "aniline": "Nc1ccccc1",
    "triphenylphosphine": "P(c1ccccc1)(c1ccccc1)c1ccccc1",
    "phenol": "Oc1ccccc1"
}

REACTION_SCHEMES = {
    "suzuki": {
        "smarts": "Brc1ccccc1.B(c1ccccc1)(O)O>>c1ccc(-c2ccccc2)cc1",
        "label": "Suzuki-Miyaura Coupling",
        "conditions": "Pd catalyst, Base"
    },
    "heck": {
        "smarts": "Brc1ccccc1.C=CC>>c1ccc(C=CC)cc1",
        "label": "Heck Reaction",
        "conditions": "Pd catalyst, Base"
    },
    "sonogashira": {
        "smarts": "Brc1ccccc1.C#Cc1ccccc1>>c1ccc(C#Cc2ccccc2)cc1",
        "label": "Sonogashira Coupling",
        "conditions": "Pd/Cu catalyst, Amine base"
    },
    "buchwald": {
        "smarts": "Brc1ccccc1.Nc1ccccc1>>c1ccc(Nc2ccccc2)cc1",
        "label": "Buchwald-Hartwig Amination",
        "conditions": "Pd catalyst, Ligand, Base"
    },
    "stille": {
        "smarts": "Brc1ccccc1.CCCC[Sn](CCCC)(CCCC)c1ccccc1>>c1ccc(-c2ccccc2)cc1",
        "label": "Stille Coupling",
        "conditions": "Pd catalyst"
    },
    "negishi": {
        "smarts": "Brc1ccccc1.[Zn](C)c1ccccc1>>c1ccc(-c2ccccc2)cc1",
        "label": "Negishi Coupling",
        "conditions": "Pd/Ni catalyst"
    },
    "kumada": {
        "smarts": "Brc1ccccc1.[Mg](C)c1ccccc1>>c1ccc(-c2ccccc2)cc1",
        "label": "Kumada Coupling",
        "conditions": "Ni/Pd catalyst"
    },
    "hiyama": {
        "smarts": "Brc1ccccc1.C[Si](C)(C)c1ccccc1>>c1ccc(-c2ccccc2)cc1",
        "label": "Hiyama Coupling",
        "conditions": "Pd catalyst, Fluoride source"
    },
    "catalytic_cycle": {
        "smarts": "Brc1ccccc1.B(c1ccccc1)(O)O>[Pd]>c1ccc(-c2ccccc2)cc1",
        "label": "Cross-Coupling Catalytic Cycle",
        "conditions": "Pd(0) → Pd(II) → Pd(0)"
    },
    "beta_hydride": {
        "smarts": "CCCPdBr>>C=CPdHBr",
        "label": "Beta-Hydride Elimination",
        "conditions": "Undesired side reaction"
    },
    "grignard": {
        "smarts": "C=O.[Mg]BrCc1ccccc1>>CC(O)c1ccccc1",
        "label": "Grignard Addition",
        "conditions": "Addition to carbonyl"
    },
    "nickel": {
        "smarts": "Clc1ccccc1.B(c1ccccc1)(O)O>[Ni]>c1ccc(-c2ccccc2)cc1",
        "label": "Ni-Catalyzed Coupling",
        "conditions": "Ni catalyst, C-Cl activation"
    },
    "triflate": {
        "smarts": "C=COS(=O)(=O)C(F)(F)F.B(c1ccccc1)(O)O>>C=Cc1ccccc1",
        "label": "Vinyl Triflate Coupling",
        "conditions": "Pd catalyst, Excellent leaving group"
    }
}

def generate_base64_mol(smiles: str) -> Optional[str]:
    if not RDKIT_AVAILABLE:
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        mol_block = Chem.MolToMolBlock(mol)
        return base64.b64encode(mol_block.encode('utf-8')).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating molecule: {e}")
        return None

def generate_reaction_image(reaction_smarts: str, width: int = 600, height: int = 150) -> Optional[str]:
    if not RDKIT_AVAILABLE:
        return None

    try:
        rxn = AllChem.ReactionFromSmarts(reaction_smarts, useSmiles=True)
        if not rxn:
            return None

        img = Draw.ReactionToImage(rxn, (width, height))
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error generating reaction: {e}")
        return None

@chemistry_bp.route('/molecules')
def get_molecules():
    data = {}
    for key, smiles in MOLECULES.items():
        b64_data = generate_base64_mol(smiles)
        if b64_data:
            data[key] = b64_data

    return jsonify({
        'success': True,
        'data': data,
        'count': len(data),
        'rdkit_available': RDKIT_AVAILABLE
    })

@chemistry_bp.route('/reactions')
def get_reactions():
    data = {}
    for key, rxn_info in REACTION_SCHEMES.items():
        img_base64 = generate_reaction_image(rxn_info["smarts"])
        if img_base64:
            data[key] = {
                "image": img_base64,
                "label": rxn_info["label"],
                "conditions": rxn_info["conditions"]
            }

    return jsonify({
        'success': True,
        'data': data,
        'count': len(data),
        'rdkit_available': RDKIT_AVAILABLE
    })

@chemistry_bp.route('/reaction/<reaction_key>')
def get_single_reaction(reaction_key: str):
    if reaction_key not in REACTION_SCHEMES:
        raise NotFoundError(f"Reaction '{reaction_key}' not found")

    rxn_info = REACTION_SCHEMES[reaction_key]
    img_base64 = generate_reaction_image(rxn_info["smarts"])

    if img_base64:
        return jsonify({
            'success': True,
            "image": img_base64,
            "label": rxn_info["label"],
            "conditions": rxn_info["conditions"]
        })

    raise APIError("Failed to generate reaction diagram", 500)
