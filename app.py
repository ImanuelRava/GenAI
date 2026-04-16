import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
import base64

# Import your citation modules
from modules.Forward_Reference import build_forward_network
from modules.Local_Reference import build_reference_network
from modules.Cross_Reference import build_cross_reference_network

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==================== STATIC PAGES ====================

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    # Handle files in root and subdirectories
    return send_from_directory('static', filename)

# ==================== CITATION NETWORK API ====================

def log_progress(msg):
    print(f"[PROGRESS]: {msg}")

@app.route('/api/network', methods=['POST'])
def analyze_network():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    analysis_type = request.form.get('type')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        G = None
        suggestions = []
        all_papers = []
        
        try:
            if analysis_type == "forward":
                G, suggestions, all_papers = build_forward_network(filepath, progress_callback=log_progress)
            elif analysis_type == "backward":
                G, suggestions, all_papers = build_reference_network(filepath, progress_callback=log_progress)
            elif analysis_type == "cross":
                G = build_cross_reference_network(filepath, progress_callback=log_progress)
                all_papers = []
                for n in G.nodes():
                    data = G.nodes[n]
                    all_papers.append({
                        'Number': len(all_papers) + 1,
                        'DOI': n,
                        'Title': data.get('title', 'No Title'),
                        'Publication Year': data.get('year', 0),
                        'Corresponding Author': data.get('author', 'Unknown'),
                        'Global Citation Count': data.get('citations', 0),
                        'Local Citation Count': G.in_degree(n)
                    })
                suggestions = []
            else:
                return jsonify({'error': 'Invalid analysis type'}), 400

            if G is None or G.number_of_nodes() < 2:
                return jsonify({'error': 'Could not build network'}), 400

            graph_json = nx.node_link_data(G, edges="edges")

            return jsonify({
                'elements': graph_json, 
                'suggestions': suggestions,
                'all_papers': all_papers,
                'stats': {
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges()
                }
            })

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
        
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

# ==================== QUIZ API (from quiz_server.py) ====================

def generate_base64_mol(smiles):
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
        print(f"Error generating molecule: {e}")
        return None

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

# ==================== RUN ====================

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)