from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from rdkit import Chem
from rdkit.Chem import AllChem
import base64

app = Flask(__name__, 
            template_folder='../frontend/pages', 
            static_folder='../frontend/pages')

CORS(app) 

def generate_base64_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
        mol_block = Chem.MolToMolBlock(mol)
        return base64.b64encode(mol_block.encode('utf-8')).decode('utf-8')
    except Exception as e:
        print(f"Error generating molecule: {e}")
        return None

@app.route('/')
def home():
    return send_from_directory('../frontend/pages', 'quiz.html')

@app.route('/api/molecules')
def get_molecules():
    molecules = {
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
    
    data = {}
    for key, smiles in molecules.items():
        b64_data = generate_base64_mol(smiles)
        if b64_data:
            data[key] = b64_data
            
    return jsonify(data)

if __name__ == '__main__':
    print("Quiz Server running on http://127.0.0.1:5001")
    app.run(debug=True, port=5001)
