import os
import sys

# Get the directory where this app.py is located (backend folder)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (download folder) for static files
BASE_DIR = os.path.dirname(BACKEND_DIR)

# Load .env file if it exists (for local development)
env_paths = [
    os.path.join(BASE_DIR, '.env'),
    os.path.join(BACKEND_DIR, '.env'),
    '/home/z/my-project/.env',
    os.path.expanduser('~/.env')
]

for env_path in env_paths:
    if os.path.exists(env_path):
        print(f"[STARTUP] Loading environment from: {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key and value and key not in os.environ:
                        os.environ[key] = value
                        # Hide API keys in logs
                        if 'KEY' in key or 'SECRET' in key:
                            print(f"  {key}=***")
                        else:
                            print(f"  {key}={value[:20]}...")
        break

# Add backend directory to path for module imports
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import networkx as nx
import base64
import io

# RDKit is optional - only needed for molecular visualization
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("[WARNING] RDKit not installed. Molecular visualization features disabled.")

# Import your citation modules
from modules.Forward_Reference import build_forward_network
from modules.Local_Reference import build_reference_network
from modules.Cross_Reference import build_cross_reference_network

# Import visualization modules
from modules.gnn_viz import (
    generate_sample_graph,
    simulate_message_passing,
    get_molecule_data,
    get_gnn_embedding_demo
)
from modules.pca_viz import (
    generate_2d_data,
    generate_scree_data,
    get_chemistry_pca_data
)

# Paths - all absolute, works from any directory
STATIC_FOLDER = BASE_DIR
UPLOAD_FOLDER = os.path.join(BACKEND_DIR, 'uploads')

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='')
CORS(app)

# Configure upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Print startup info
print(f"[STARTUP] Backend Dir: {BACKEND_DIR}")
print(f"[STARTUP] Static Folder: {STATIC_FOLDER}")
print(f"[STARTUP] Upload Folder: {UPLOAD_FOLDER}")

# ==================== STATIC PAGES ====================

@app.route('/')
def index():
    return send_from_directory(STATIC_FOLDER, 'index.html')

@app.route('/TMC/')
def TMC_index():
    return send_from_directory(os.path.join(STATIC_FOLDER, 'TMC'), 'index.html')

@app.route('/TMC/<path:filename>')
def TMC_files(filename):
    return send_from_directory(os.path.join(STATIC_FOLDER, 'TMC'), filename)

@app.route('/AI/')
def AI_index():
    return send_from_directory(os.path.join(STATIC_FOLDER, 'AI'), 'index.html')

@app.route('/AI/<path:filename>')
def AI_files(filename):
    return send_from_directory(os.path.join(STATIC_FOLDER, 'AI'), filename)

@app.route('/virus/')
def virus_index():
    return send_from_directory(os.path.join(STATIC_FOLDER, 'virus'), 'index.html')

@app.route('/virus/<path:filename>')
def virus_files(filename):
    return send_from_directory(os.path.join(STATIC_FOLDER, 'virus'), filename)

@app.route('/redox-ligands/')
def redox_index():
    return send_from_directory(os.path.join(STATIC_FOLDER, 'redox-ligands'), 'index.html')

@app.route('/redox-ligands/<path:filename>')
def redox_files(filename):
    return send_from_directory(os.path.join(STATIC_FOLDER, 'redox-ligands'), filename)

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

# ==================== QUIZ API ====================

def generate_base64_mol(smiles):
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


# ==================== REACTION DIAGRAM API ====================

def generate_reaction_image(reaction_smarts, width=600, height=150):
    """Generate a reaction diagram image using RDKit"""
    if not RDKIT_AVAILABLE:
        return None
    try:
        rxn = AllChem.ReactionFromSmarts(reaction_smarts, useSmiles=True)
        if not rxn:
            return None
        
        # Generate reaction image
        img = Draw.ReactionToImage(rxn, (width, height))
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_base64
    except Exception as e:
        print(f"Error generating reaction: {e}")
        return None


# Define reactions with SMILES/SMARTS
REACTION_SCHEMES = {
    # Suzuki: Ar-X + Ar'-B(OH)2 -> Ar-Ar'
    "suzuki": {
        "smarts": "Brc1ccccc1.B(c1ccccc1)(O)O>>c1ccc(-c2ccccc2)cc1",
        "label": "Suzuki-Miyaura Coupling",
        "conditions": "Pd catalyst, Base"
    },
    # Heck: Ar-X + Alkene -> Ar-alkene
    "heck": {
        "smarts": "Brc1ccccc1.C=CC>>c1ccc(C=CC)cc1",
        "label": "Heck Reaction",
        "conditions": "Pd catalyst, Base"
    },
    # Sonogashira: Ar-X + Alkyne -> Ar-alkyne
    "sonogashira": {
        "smarts": "Brc1ccccc1.C#Cc1ccccc1>>c1ccc(C#Cc2ccccc2)cc1",
        "label": "Sonogashira Coupling",
        "conditions": "Pd/Cu catalyst, Amine base"
    },
    # Buchwald-Hartwig: Ar-X + Amine -> Ar-NR2
    "buchwald": {
        "smarts": "Brc1ccccc1.Nc1ccccc1>>c1ccc(Nc2ccccc2)cc1",
        "label": "Buchwald-Hartwig Amination",
        "conditions": "Pd catalyst, Ligand, Base"
    },
    # Stille: Ar-X + R-SnBu3 -> Ar-R
    "stille": {
        "smarts": "Brc1ccccc1.CCCC[Sn](CCCC)(CCCC)c1ccccc1>>c1ccc(-c2ccccc2)cc1",
        "label": "Stille Coupling",
        "conditions": "Pd catalyst"
    },
    # Negishi: Ar-X + R-ZnX -> Ar-R
    "negishi": {
        "smarts": "Brc1ccccc1.[Zn](C)c1ccccc1>>c1ccc(-c2ccccc2)cc1",
        "label": "Negishi Coupling",
        "conditions": "Pd/Ni catalyst"
    },
    # Kumada: Ar-X + R-MgX -> Ar-R
    "kumada": {
        "smarts": "Brc1ccccc1.[Mg](C)c1ccccc1>>c1ccc(-c2ccccc2)cc1",
        "label": "Kumada Coupling",
        "conditions": "Ni/Pd catalyst"
    },
    # Hiyama: Ar-X + R-SiR3 -> Ar-R
    "hiyama": {
        "smarts": "Brc1ccccc1.C[Si](C)(C)c1ccccc1>>c1ccc(-c2ccccc2)cc1",
        "label": "Hiyama Coupling",
        "conditions": "Pd catalyst, Fluoride source"
    },
    # Grignard addition: R-MgX + Aldehyde -> Alcohol
    "grignard": {
        "smarts": "[Mg](C)c1ccccc1.C=O>>OC(-c1ccccc1)c1ccccc1",
        "label": "Grignard Addition",
        "conditions": "Anhydrous conditions"
    },
    # Oxidative addition step
    "oxidative_addition": {
        "smarts": "Brc1ccccc1.[Pd]>>[Pd](Br)(-c1ccccc1)",
        "label": "Oxidative Addition",
        "conditions": "Pd(0) → Pd(II)"
    },
    # Catalytic cycle simplified
    "catalytic_cycle": {
        "smarts": "Brc1ccccc1.B(c1ccccc1)(O)O>>c1ccc(-c2ccccc2)cc1",
        "label": "Pd-Catalyzed Cross-Coupling",
        "conditions": "Catalytic Cycle"
    },
    # Beta-hydride elimination
    "beta_hydride": {
        "smarts": "[Pd](-C-C-C)>>[Pd]H.C=CC",
        "label": "β-Hydride Elimination",
        "conditions": "Side Reaction"
    },
    # Vinyl triflate coupling
    "triflate": {
        "smarts": "C=COS(=O)(=O)C(F)(F)F.B(c1ccccc1)(O)O>>C=C-c1ccccc1",
        "label": "Vinyl Triflate Coupling",
        "conditions": "Pd catalyst"
    },
    # Nickel catalyzed
    "nickel": {
        "smarts": "Clc1ccccc1.B(c1ccccc1)(O)O>>c1ccc(-c2ccccc2)cc1",
        "label": "Ni-Catalyzed Coupling",
        "conditions": "Ni activates C-Cl bonds"
    }
}

@app.route('/api/reactions')
def get_reactions():
    """Return all reaction diagrams as base64 PNG images"""
    data = {}
    for key, rxn_info in REACTION_SCHEMES.items():
        img_base64 = generate_reaction_image(rxn_info["smarts"])
        if img_base64:
            data[key] = {
                "image": img_base64,
                "label": rxn_info["label"],
                "conditions": rxn_info["conditions"]
            }
    return jsonify(data)


@app.route('/api/reaction/<reaction_key>')
def get_single_reaction(reaction_key):
    """Return a single reaction diagram"""
    if reaction_key not in REACTION_SCHEMES:
        return jsonify({"error": "Reaction not found"}), 404
    
    rxn_info = REACTION_SCHEMES[reaction_key]
    img_base64 = generate_reaction_image(rxn_info["smarts"])
    
    if img_base64:
        return jsonify({
            "image": img_base64,
            "label": rxn_info["label"],
            "conditions": rxn_info["conditions"]
        })
    return jsonify({"error": "Failed to generate reaction"}), 500


# ==================== GNN VISUALIZATION API ====================

@app.route('/api/gnn/graph')
def api_gnn_graph():
    """Generate a sample graph for GNN visualization"""
    try:
        num_nodes = request.args.get('nodes', 6, type=int)
        data = generate_sample_graph(num_nodes)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/gnn/message-passing', methods=['POST'])
def api_gnn_message_passing():
    """Simulate one step of message passing"""
    try:
        data = request.get_json()
        nodes = data.get('nodes', [])
        edges = data.get('edges', [])
        current_step = data.get('currentStep', 0)
        
        result = simulate_message_passing(nodes, edges, current_step)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/gnn/molecule/<molecule_type>')
def api_gnn_molecule(molecule_type):
    """Get molecular graph data for visualization"""
    try:
        data = get_molecule_data(molecule_type)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/gnn/embeddings')
def api_gnn_embeddings():
    """Get GNN embedding demo data"""
    try:
        data = get_gnn_embedding_demo()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== PCA VISUALIZATION API ====================

@app.route('/api/pca/data/<data_type>')
def api_pca_data(data_type):
    """Generate 2D data for PCA projection visualization"""
    try:
        n_samples = request.args.get('samples', 60, type=int)
        data = generate_2d_data(data_type, n_samples)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pca/scree')
def api_pca_scree():
    """Generate scree plot data"""
    try:
        num_features = request.args.get('features', 10, type=int)
        data_type = request.args.get('type', 'structured')
        data = generate_scree_data(num_features, data_type)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pca/chemistry/<dataset>')
def api_pca_chemistry(dataset):
    """Get chemistry dataset PCA visualization"""
    try:
        data = get_chemistry_pca_data(dataset)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== LLM KNOWLEDGE GRAPH API ====================

# Import pure Python LLM providers (works on PythonAnywhere without Node.js)
from llm_providers import get_llm_response, generate_knowledge_graph as llm_generate_kg, explain_concept

def get_llm_response_sync(system_prompt: str, user_message: str) -> str:
    """Get response from LLM using pure Python providers"""
    return get_llm_response(system_prompt, user_message)


def generate_knowledge_graph_with_llm(topic: str) -> dict:
    """Use LLM to generate a knowledge graph for a given topic"""
    # Use the pure Python LLM provider
    return llm_generate_kg(topic)


def generate_mock_knowledge_graph(topic: str) -> dict:
    """Generate a mock knowledge graph for TMC topics"""
    topic_lower = topic.lower()
    
    # Predefined knowledge graphs for common TMC topics
    graphs = {
        'suzuki': {
            "nodes": [
                {"id": "suzuki", "label": "Suzuki-Miyaura Coupling", "type": "reaction", "description": "Pd-catalyzed cross-coupling between organoboron and organic halide"},
                {"id": "palladium", "label": "Palladium Catalyst", "type": "catalyst", "description": "Transition metal catalyst essential for Suzuki reaction"},
                {"id": "boronic", "label": "Organoboron Reagent", "type": "reagent", "description": "R-B(OH)2 nucleophilic partner"},
                {"id": "halide", "label": "Organic Halide", "type": "reagent", "description": "R'-X electrophilic partner (X = Br, I, Cl)"},
                {"id": "base", "label": "Base", "type": "reagent", "description": "Required for transmetalation step"},
                {"id": "oxidative", "label": "Oxidative Addition", "type": "mechanism", "description": "Pd(0) → Pd(II), inserts into C-X bond"},
                {"id": "transmetalation", "label": "Transmetalation", "type": "mechanism", "description": "Transfer of R group from boron to Pd"},
                {"id": "reductive", "label": "Reductive Elimination", "type": "mechanism", "description": "Pd(II) → Pd(0), forms C-C bond"},
                {"id": "biaryl", "label": "Biaryl Product", "type": "product", "description": "R-R' coupled product"}
            ],
            "edges": [
                {"source": "suzuki", "target": "palladium", "label": "catalyzed by"},
                {"source": "suzuki", "target": "boronic", "label": "uses"},
                {"source": "suzuki", "target": "halide", "label": "uses"},
                {"source": "suzuki", "target": "base", "label": "requires"},
                {"source": "palladium", "target": "oxidative", "label": "undergoes"},
                {"source": "oxidative", "target": "transmetalation", "label": "followed by"},
                {"source": "transmetalation", "target": "reductive", "label": "followed by"},
                {"source": "reductive", "target": "biaryl", "label": "produces"},
                {"source": "boronic", "target": "transmetalation", "label": "participates in"},
                {"source": "halide", "target": "oxidative", "label": "participates in"}
            ]
        },
        'heck': {
            "nodes": [
                {"id": "heck", "label": "Heck Reaction", "type": "reaction", "description": "Pd-catalyzed coupling of aryl halide with alkene"},
                {"id": "palladium", "label": "Palladium Catalyst", "type": "catalyst", "description": "Pd(0)/Pd(II) catalytic cycle"},
                {"id": "aryl_halide", "label": "Aryl Halide", "type": "reagent", "description": "Ar-X electrophile"},
                {"id": "alkene", "label": "Alkene", "type": "reagent", "description": "C=C nucleophilic partner"},
                {"id": "migratory", "label": "Migratory Insertion", "type": "mechanism", "description": "Alkene inserts into Pd-Ar bond"},
                {"id": "beta_hydride", "label": "β-Hydride Elimination", "type": "mechanism", "description": "Forms substituted alkene product"},
                {"id": "styrene", "label": "Styrene Derivative", "type": "product", "description": "Ar-CH=CH2 type product"}
            ],
            "edges": [
                {"source": "heck", "target": "palladium", "label": "catalyzed by"},
                {"source": "heck", "target": "aryl_halide", "label": "uses"},
                {"source": "heck", "target": "alkene", "label": "uses"},
                {"source": "palladium", "target": "migratory", "label": "undergoes"},
                {"source": "migratory", "target": "beta_hydride", "label": "followed by"},
                {"source": "beta_hydride", "target": "styrene", "label": "produces"},
                {"source": "alkene", "target": "migratory", "label": "participates in"}
            ]
        },
        'nickel': {
            "nodes": [
                {"id": "nickel", "label": "Nickel Catalysis", "type": "catalyst", "description": "Cost-effective alternative to Pd, activates C-Cl and C-O bonds"},
                {"id": "c_o_activation", "label": "C-O Bond Activation", "type": "mechanism", "description": "Ni can break strong C-O bonds in biomass"},
                {"id": "c_cl_activation", "label": "C-Cl Bond Activation", "type": "mechanism", "description": "Ni activates aryl chlorides efficiently"},
                {"id": "biomass", "label": "Biomass Derivatives", "type": "reagent", "description": "Sustainable feedstocks (phenols, ethers)"},
                {"id": "redox_active", "label": "Redox-Active Ligands", "type": "ligand", "description": "Ligands that participate in electron transfer"},
                {"id": "low_cost", "label": "Cost Advantage", "type": "property", "description": "Ni is ~1000x cheaper than Pd"},
                {"id": "oxidation_states", "label": "Multiple Oxidation States", "type": "property", "description": "Ni(0), Ni(I), Ni(II), Ni(III) accessible"}
            ],
            "edges": [
                {"source": "nickel", "target": "c_o_activation", "label": "enables"},
                {"source": "nickel", "target": "c_cl_activation", "label": "enables"},
                {"source": "c_o_activation", "target": "biomass", "label": "activates"},
                {"source": "nickel", "target": "redox_active", "label": "benefits from"},
                {"source": "nickel", "target": "low_cost", "label": "has"},
                {"source": "nickel", "target": "oxidation_states", "label": "accesses"}
            ]
        }
    }
    
    # Default cross-coupling knowledge graph
    default_graph = {
        "nodes": [
            {"id": "cross_coupling", "label": "Cross-Coupling", "type": "reaction", "description": "Metal-catalyzed C-C bond formation"},
            {"id": "oxidative_addition", "label": "Oxidative Addition", "type": "mechanism", "description": "M(0) → M(II), inserts into C-X bond"},
            {"id": "transmetalation", "label": "Transmetalation", "type": "mechanism", "description": "Exchange of ligands between metals"},
            {"id": "reductive_elimination", "label": "Reductive Elimination", "type": "mechanism", "description": "M(II) → M(0), forms product"},
            {"id": "palladium", "label": "Palladium", "type": "catalyst", "description": "Most common cross-coupling catalyst"},
            {"id": "nickel", "label": "Nickel", "type": "catalyst", "description": "Cheaper alternative, activates C-Cl/C-O"},
            {"id": "ligand", "label": "Ligand", "type": "ligand", "description": "Controls reactivity and selectivity"},
            {"id": "base", "label": "Base", "type": "reagent", "description": "Often required for transmetalation"},
            {"id": "product", "label": "C-C Bond Product", "type": "product", "description": "Coupled organic molecule"}
        ],
        "edges": [
            {"source": "cross_coupling", "target": "oxidative_addition", "label": "step 1"},
            {"source": "oxidative_addition", "target": "transmetalation", "label": "step 2"},
            {"source": "transmetalation", "target": "reductive_elimination", "label": "step 3"},
            {"source": "reductive_elimination", "target": "product", "label": "produces"},
            {"source": "cross_coupling", "target": "palladium", "label": "catalyzed by"},
            {"source": "cross_coupling", "target": "nickel", "label": "catalyzed by"},
            {"source": "palladium", "target": "ligand", "label": "requires"},
            {"source": "nickel", "target": "ligand", "label": "requires"},
            {"source": "transmetalation", "target": "base", "label": "may require"}
        ]
    }
    
    # Match topic to predefined graph
    for key, graph in graphs.items():
        if key in topic_lower:
            return graph
    
    return default_graph


@app.route('/api/knowledge-graph', methods=['POST'])
def api_knowledge_graph():
    """Generate knowledge graph for a given topic using LLM"""
    try:
        data = request.get_json()
        topic = data.get('topic', 'cross-coupling')
        use_llm = data.get('use_llm', True)  # Default to using LLM

        print(f"[KG API] Topic: {topic}, Use LLM: {use_llm}")

        graph_data = None
        llm_used = False

        if use_llm:
            # Try to use LLM to generate knowledge graph
            print(f"[KG API] Attempting LLM generation for: {topic}")
            graph_data = generate_knowledge_graph_with_llm(topic)
            if graph_data:
                llm_used = True
                print(f"[KG API] LLM generation successful, {len(graph_data.get('nodes', []))} nodes")

        if not graph_data:
            # Fallback to mock knowledge graph
            print(f"[KG API] Using mock data for: {topic}")
            graph_data = generate_mock_knowledge_graph(topic)

        return jsonify({
            'success': True,
            'topic': topic,
            'graph': graph_data,
            'llm_used': llm_used
        })

    except Exception as e:
        print(f"Knowledge Graph Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/knowledge-graph/explain', methods=['POST'])
def api_knowledge_graph_explain():
    """Get LLM explanation for a node or relationship"""
    try:
        data = request.get_json()
        node_label = data.get('node', '')
        context = data.get('context', '')

        # Try LLM explanation first
        system_prompt = """You are an expert chemistry educator specializing in transition metal catalysis.
Provide a clear, concise explanation (2-3 sentences) for the given chemistry concept.
Focus on practical understanding and real-world applications.
Keep the explanation accessible to graduate-level chemistry students."""

        user_message = f"Explain {node_label} in the context of transition metal catalysis. Context: {context}"

        llm_response = get_llm_response_sync(system_prompt, user_message)

        if llm_response:
            return jsonify({
                'success': True,
                'node': node_label,
                'explanation': llm_response,
                'source': 'llm'
            })

        # Fallback to predefined explanations
        explanations = {
            'oxidative addition': 'Oxidative addition is the first step in cross-coupling. The metal catalyst (M) inserts into the C-X bond of the organic halide. The metal oxidation state increases by 2 (e.g., Pd(0) → Pd(II)) as it forms two new bonds.',
            'transmetalation': 'Transmetalation is the transfer of an organic group from the nucleophilic reagent (R-M) to the metal center. This step pairs the two organic fragments on the metal before coupling.',
            'reductive elimination': 'Reductive elimination is the final step where the two organic groups couple together and are released as the product. The metal is reduced back to its original oxidation state (e.g., Pd(II) → Pd(0)).',
            'palladium': 'Palladium is the most widely used catalyst for cross-coupling reactions. Pd(0) complexes are nucleophilic and readily undergo oxidative addition. The 2010 Nobel Prize was awarded for Pd-catalyzed cross-couplings.',
            'nickel': 'Nickel is a cost-effective alternative to palladium. Ni is more electrophilic and can activate stronger bonds like C-Cl and C-O. This makes it valuable for sustainable chemistry using biomass-derived feedstocks.',
            'suzuki': 'Suzuki-Miyaura coupling uses organoboron reagents. Key advantages: non-toxic, air-stable reagents, aqueous compatible. Won the 2010 Nobel Prize (Suzuki).',
            'heck': 'Heck reaction couples aryl halides with alkenes. Unique in that it does not require an organometallic nucleophile. Products are substituted alkenes.',
            'ligand': 'Ligands control the reactivity, selectivity, and stability of metal catalysts. Electron-rich ligands favor oxidative addition, while bulky ligands prevent unwanted side reactions.'
        }

        explanation = explanations.get(node_label.lower(),
            f'{node_label} is an important concept in transition metal catalysis. It plays a crucial role in the catalytic cycle and influences reaction outcomes.')

        return jsonify({
            'success': True,
            'node': node_label,
            'explanation': explanation,
            'source': 'predefined'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


# ==================== PDF UPLOAD & KNOWLEDGE GRAPH API ====================

import pdfplumber

def extract_text_from_pdf(filepath: str) -> str:
    """Extract text content from a PDF file"""
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return None
    return text.strip()


def generate_kg_from_content(content: str, source_name: str = "PDF") -> dict:
    """Generate knowledge graph from text content using LLM"""
    
    # Truncate content if too long (LLM token limits)
    max_chars = 6000  # Reduced to ensure complete response
    if len(content) > max_chars:
        content = content[:max_chars] + "..."
    
    system_prompt = """You are a knowledge extraction expert. Extract key concepts from the text and create a knowledge graph.

IMPORTANT: Return ONLY valid JSON with no markdown, no code blocks, no explanation.

Return a JSON object with this EXACT structure:
{"nodes":[{"id":"id","label":"Name","type":"concept","description":"desc"}],"edges":[{"source":"id","target":"id","label":"rel"}]}

Node types: concept, reaction, catalyst, reagent, mechanism, product, method, theory, property

Rules:
- Extract only 8-12 most important concepts
- Use short IDs in snake_case
- Keep descriptions under 15 words
- Make sure ALL JSON brackets and quotes are properly closed
- Return ONLY the JSON object, nothing else"""

    user_message = f"Extract knowledge graph from this text:\n\n{content}"

    response = get_llm_response_sync(system_prompt, user_message)
    
    if response:
        try:
            # Clean up response - remove markdown code blocks
            json_str = response.strip()
            
            # Remove markdown code blocks if present
            if '```' in json_str:
                # Extract content between code blocks
                lines = json_str.split('\n')
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block or not line.strip().startswith('```'):
                        json_lines.append(line)
                json_str = '\n'.join(json_lines)
            
            # Find JSON object boundaries
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = json_str[start_idx:end_idx + 1]
            
            import json
            result = json.loads(json_str)
            
            # Validate structure
            if 'nodes' in result and 'edges' in result:
                return result
            else:
                print(f"Invalid structure: missing nodes or edges")
                return None
                
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Attempting to fix incomplete JSON...")
            
            # Try to fix incomplete JSON
            try:
                fixed_json = fix_incomplete_json(json_str)
                if fixed_json:
                    return fixed_json
            except Exception as fix_error:
                print(f"JSON fix failed: {fix_error}")
            
            return None
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    return None


def fix_incomplete_json(json_str: str) -> dict:
    """Attempt to fix incomplete JSON by closing open brackets"""
    import json
    
    # Count brackets
    open_braces = json_str.count('{') - json_str.count('}')
    open_brackets = json_str.count('[') - json_str.count(']')
    
    # Add missing closing brackets
    fixed = json_str
    for _ in range(open_brackets):
        fixed += ']'
    for _ in range(open_braces):
        fixed += '}'
    
    try:
        result = json.loads(fixed)
        if 'nodes' in result:
            # Clean up any incomplete nodes
            if isinstance(result.get('nodes'), list):
                result['nodes'] = [n for n in result['nodes'] if isinstance(n, dict) and 'id' in n]
            if isinstance(result.get('edges'), list):
                result['edges'] = [e for e in result['edges'] if isinstance(e, dict) and 'source' in e and 'target' in e]
            return result
    except:
        pass
    
    return None


@app.route('/api/knowledge-graph/upload', methods=['POST'])
def api_knowledge_graph_upload():
    """Upload PDF and generate knowledge graph from its content"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded', 'success': False}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        # Check file extension
        filename = file.filename.lower()
        if not filename.endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported', 'success': False}), 400
        
        # Save uploaded file temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        print(f"[KG Upload] Processing PDF: {file.filename}")
        
        try:
            # Extract text from PDF
            text_content = extract_text_from_pdf(filepath)
            
            if not text_content:
                return jsonify({
                    'error': 'Could not extract text from PDF. The PDF may be scanned or image-based.',
                    'success': False
                }), 400
            
            print(f"[KG Upload] Extracted {len(text_content)} characters from PDF")
            
            # Generate knowledge graph using LLM
            graph_data = generate_kg_from_content(text_content, file.filename)
            
            if not graph_data:
                return jsonify({
                    'error': 'Failed to generate knowledge graph from PDF content',
                    'success': False
                }), 500
            
            # Extract title/topic from content
            lines = text_content.split('\n')
            title = file.filename.replace('.pdf', '').replace('_', ' ')
            
            print(f"[KG Upload] Generated graph with {len(graph_data.get('nodes', []))} nodes")
            
            return jsonify({
                'success': True,
                'topic': title,
                'graph': graph_data,
                'llm_used': True,
                'content_length': len(text_content),
                'source': 'pdf_upload'
            })
            
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        print(f"KG Upload Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


# ==================== RUN ====================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("GenAI Research Flask Server")
    print("="*50)
    print(f"Static files served from: {STATIC_FOLDER}")
    print(f"Access the site at: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
