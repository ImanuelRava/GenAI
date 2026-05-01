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

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import networkx as nx
import base64
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
from modules.Backward_Reference import build_reference_network
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

# Security Configuration
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

# CORS Configuration - can be set via environment variable
# For multiple origins, separate with commas: "http://localhost:3000,https://example.com"
_default_origins = 'http://localhost:5000,http://127.0.0.1:5000,http://localhost:3000'
_cors_origins = os.environ.get('CORS_ORIGINS', _default_origins)
ALLOWED_ORIGINS = [origin.strip() for origin in _cors_origins.split(',') if origin.strip()]

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='')

# Configure CORS with restricted origins
CORS(app, resources={
    r"/api/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Configure upload folder with size limit
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_file_upload(file) -> tuple:
    """Validate uploaded file and return (is_valid, error_message)"""
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, None

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
@limiter.limit("10 per minute")
def analyze_network():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    analysis_type = request.form.get('type')
    
    # Validate file upload
    is_valid, error_msg = validate_file_upload(file)
    if not is_valid:
        return jsonify({'error': error_msg}), 400
    
    # Validate analysis type
    valid_types = ['forward', 'backward', 'cross']
    if analysis_type not in valid_types:
        return jsonify({'error': f'Invalid analysis type. Must be one of: {valid_types}'}), 400

    if file:
        # Use secure filename to prevent path traversal
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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

            # Manually format graph data for maximum compatibility across NetworkX versions
            # This ensures nodes have 'id' field and edges have proper 'source'/'target' IDs
            nodes = []
            for node_id in G.nodes():
                node_data = dict(G.nodes[node_id])
                node_data['id'] = node_id  # Ensure ID is included
                # Convert is_main boolean to string for Cytoscape selector compatibility
                if 'is_main' in node_data:
                    node_data['is_main'] = "True" if node_data['is_main'] else "False"
                nodes.append(node_data)
            
            edges = []
            for source, target in G.edges():
                edges.append({
                    'source': source,
                    'target': target
                })
            
            graph_json = {
                'nodes': nodes,
                'edges': edges
            }

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
            logger.error(f"Network analysis error: {e}", exc_info=True)
            return jsonify({'error': 'An internal error occurred during analysis'}), 500
        
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
from llm_providers import get_llm_response, generate_knowledge_graph as llm_generate_kg, explain_concept, LLMProviderFactory

@app.route('/api/llm/status')
def api_llm_status():
    """Check if any LLM API key is configured on the backend"""
    providers_with_keys = []
    
    # Check for each provider's API key in environment
    if os.environ.get('GROQ_API_KEY'):
        providers_with_keys.append({'provider': 'groq', 'name': 'Groq', 'configured': True})
    if os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'):
        providers_with_keys.append({'provider': 'gemini', 'name': 'Google Gemini', 'configured': True})
    if os.environ.get('HF_API_KEY') or os.environ.get('HUGGINGFACE_API_KEY'):
        providers_with_keys.append({'provider': 'huggingface', 'name': 'Hugging Face', 'configured': True})
    if os.environ.get('DEEPSEEK_API_KEY'):
        providers_with_keys.append({'provider': 'deepseek', 'name': 'DeepSeek', 'configured': True})
    if os.environ.get('OPENROUTER_API_KEY'):
        providers_with_keys.append({'provider': 'openrouter', 'name': 'OpenRouter', 'configured': True})
    if os.environ.get('OPENAI_API_KEY'):
        providers_with_keys.append({'provider': 'openai', 'name': 'OpenAI', 'configured': True})
    if os.environ.get('ANTHROPIC_API_KEY'):
        providers_with_keys.append({'provider': 'anthropic', 'name': 'Anthropic', 'configured': True})
    
    # Get the default provider
    default_provider = LLMProviderFactory.get_default_provider()
    
    return jsonify({
        'has_backend_key': len(providers_with_keys) > 0,
        'providers': providers_with_keys,
        'default_provider': default_provider
    })

def get_llm_response_sync(system_prompt: str, user_message: str, 
                          provider: str = None, api_key: str = None) -> str:
    """Get response from LLM using pure Python providers"""
    return get_llm_response(system_prompt, user_message, provider=provider, api_key=api_key)


def generate_knowledge_graph_with_llm(topic: str, provider: str = None, api_key: str = None) -> dict:
    """Use LLM to generate a knowledge graph for a given topic"""
    # Use the pure Python LLM provider
    return llm_generate_kg(topic, provider=provider, api_key=api_key)


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
    """Generate knowledge graph for a given topic using LLM
    
    Request body:
        topic: The topic to generate knowledge graph for
        use_llm: Whether to use LLM (default: True)
        provider: LLM provider (deepseek, openai, groq, gemini, huggingface, openrouter, ollama)
        api_key: User's API key for the LLM provider
    """
    try:
        data = request.get_json()
        topic = data.get('topic', 'cross-coupling')
        use_llm = data.get('use_llm', True)  # Default to using LLM
        provider = data.get('provider')  # LLM provider (optional)
        api_key = data.get('api_key')  # User's API key (optional for ollama)

        print(f"[KG API] Topic: {topic}, Use LLM: {use_llm}, Provider: {provider}")

        graph_data = None
        llm_used = False

        if use_llm:
            # Try to use LLM to generate knowledge graph
            print(f"[KG API] Attempting LLM generation for: {topic}")
            graph_data = generate_knowledge_graph_with_llm(topic, provider=provider, api_key=api_key)
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
    """Get LLM explanation for a node or relationship
    
    Request body:
        node: The node/concept to explain
        context: Additional context
        provider: LLM provider (optional)
        api_key: User's API key (optional)
    """
    try:
        data = request.get_json()
        node_label = data.get('node', '')
        context = data.get('context', '')
        provider = data.get('provider')  # LLM provider (optional)
        api_key = data.get('api_key')  # User's API key (optional)

        # Try LLM explanation first
        system_prompt = """You are an expert chemistry educator specializing in transition metal catalysis.
Provide a clear, concise explanation (2-3 sentences) for the given chemistry concept.
Focus on practical understanding and real-world applications.
Keep the explanation accessible to graduate-level chemistry students."""

        user_message = f"Explain {node_label} in the context of transition metal catalysis. Context: {context}"

        llm_response = get_llm_response_sync(system_prompt, user_message, provider=provider, api_key=api_key)

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


def generate_kg_from_content(content: str, source_name: str = "PDF", 
                             provider: str = None, api_key: str = None) -> dict:
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

    response = get_llm_response_sync(system_prompt, user_message, provider=provider, api_key=api_key)
    
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
@limiter.limit("5 per minute")
def api_knowledge_graph_upload():
    """Upload PDF and generate knowledge graph from its content
    
    Form data:
        file: PDF file
        provider: LLM provider (optional)
        api_key: User's API key (optional)
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded', 'success': False}), 400
        
        file = request.files['file']
        provider = request.form.get('provider')  # LLM provider (optional)
        api_key = request.form.get('api_key')  # User's API key (optional)
        
        # Validate file upload
        is_valid, error_msg = validate_file_upload(file)
        if not is_valid:
            return jsonify({'error': error_msg, 'success': False}), 400
        
        # Use secure filename to prevent path traversal
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({'error': 'Invalid filename', 'success': False}), 400
        
        # Save uploaded file temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing PDF: {filename}")
        
        try:
            # Extract text from PDF
            text_content = extract_text_from_pdf(filepath)
            
            if not text_content:
                return jsonify({
                    'error': 'Could not extract text from PDF. The PDF may be scanned or image-based.',
                    'success': False
                }), 400
            
            logger.info(f"Extracted {len(text_content)} characters from PDF")
            
            # Generate knowledge graph using LLM
            graph_data = generate_kg_from_content(text_content, filename, provider=provider, api_key=api_key)
            
            if not graph_data:
                return jsonify({
                    'error': 'Failed to generate knowledge graph from PDF content',
                    'success': False
                }), 500
            
            # Extract title/topic from content
            title = filename.replace('.pdf', '').replace('_', ' ')
            
            logger.info(f"Generated graph with {len(graph_data.get('nodes', []))} nodes")
            
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
        logger.error(f"KG Upload Error: {e}", exc_info=True)
        return jsonify({'error': 'An internal error occurred', 'success': False}), 500


# ==================== REDOX-ACTIVE LIGAND CHATBOT API ====================

# Specialized knowledge base for redox-active ligands
REDOX_KNOWLEDGE = {
    "fundamentals": {
        "definition": "Redox-active ligands (also called non-innocent ligands) are ligands that can undergo reversible electron transfer reactions, participating directly in the redox chemistry of metal complexes. Unlike innocent ligands that maintain constant oxidation states, redox-active ligands can exist in multiple oxidation states and act as electron reservoirs.",
        "innocent_vs_noninnocent": "Innocent ligands (like phosphines, amines) do not change oxidation state during reactions - all redox changes occur at the metal center. Non-innocent (redox-active) ligands can accept or donate electrons, participating in the redox process. This distinction was first articulated by Jørgensen and Cotton.",
        "metal_ligand_cooperativity": "Metal-ligand cooperativity (MLC) describes systems where both the metal and ligand participate in bond-making and bond-breaking events. Redox-active ligands enable two-electron processes at metals that typically undergo one-electron changes, expanding accessible reactivity patterns."
    },
    "ligand_classes": {
        "pdi": {
            "name": "Bis(imino)pyridine (PDI)",
            "description": "One of the most studied redox-active ligand families. The diimine backbone can accept up to 2 electrons, forming radical anion and dianion states. Key for nickel-catalyzed C-O activation reactions.",
            "redox_states": ["Neutral (PDI)", "Radical anion (PDI•−)", "Dianion (PDI2−)"],
            "applications": ["C-O bond activation", "Cross-coupling catalysis", "Small molecule activation (N2, CO2, H2)"]
        },
        "catecholate": {
            "name": "Catecholate/o-Quinone",
            "description": "Reversible two-electron redox couple between catecholate (dianion) and o-quinone (neutral). Can stabilize high-valent metal centers. Found in nature (e.g., copper amine oxidase).",
            "redox_states": ["Catecholate (Cat2−)", "Semiquinone (SQ•−)", "o-Quinone (Q)"],
            "applications": ["Oxygen activation", "Electron transfer in biology", "Metalloenzyme mimics"]
        },
        "dithiolene": {
            "name": "Dithiolene Ligands",
            "description": "Sulfur-based ligands with extensive redox activity. Can form stable complexes with metals in unusual oxidation states. Important in molybdenum enzymes.",
            "redox_states": ["Enedithiolate", "Dithiolene radical", "Dithione"],
            "applications": ["Mo/W enzymes", "Conductive materials", "Solar energy conversion"]
        },
        "alpha_diimine": {
            "name": "α-Diimine Ligands",
            "description": "Include bipyridine, phenanthroline derivatives. Can undergo one-electron reduction to form radical anions. Widely used in photoredox catalysis and electrochemistry.",
            "redox_states": ["Neutral diimine", "Radical anion", "Dianion"],
            "applications": ["Photoredox catalysis", "Electrocatalysis", "CO2 reduction"]
        },
        "redox_nhc": {
            "name": "Redox-Active N-Heterocyclic Carbenes",
            "description": "NHCs with redox-active substituents (ferrocene, quinone) or backbone modifications. Combine strong σ-donation with redox activity.",
            "redox_states": "Depends on substituent",
            "applications": ["Tunable catalysis", "Electrochemical switches", "Stabilization of reactive intermediates"]
        }
    },
    "mechanisms": {
        "electron_reservoir": "Redox-active ligands act as electron reservoirs, storing electrons that can be delivered to substrates. This enables metal centers to access oxidation states they couldn't reach alone. For example, Ni(II) can effectively perform two-electron reductions when paired with a reducible ligand.",
        "two_electron_processes": "First-row transition metals (Ni, Fe, Co) typically undergo one-electron redox changes. Redox-active ligands enable these metals to mimic noble metal behavior by storing the second electron, facilitating two-electron processes like oxidative addition/reductive elimination.",
        "substrate_activation": "The stored electrons in redox-active ligands can be transferred to substrates directly. This is crucial for activating inert bonds (C-O, C-F, C-H) and small molecules (N2, CO2, O2). The ligand acts as a 'redox buffer' during catalytic cycles."
    },
    "applications": {
        "nickel_catalysis": "Redox-active ligands are particularly important for nickel catalysis. Ni is ~1000x cheaper than Pd but traditionally limited by one-electron redox chemistry. Ligands like PDI enable Ni to catalyze cross-couplings (Suzuki, Heck) and C-O activation reactions previously requiring Pd.",
        "c_o_activation": "C-O bond activation in aryl ethers and esters is challenging due to strong bonds (~85-90 kcal/mol). Ni-PDI complexes can cleave these bonds, enabling conversion of biomass-derived compounds (lignin model compounds) into valuable chemicals.",
        "cross_coupling": "Redox-active ligands expand cross-coupling capabilities: (1) Enable use of cheaper metals (Ni, Fe instead of Pd), (2) Allow activation of stronger bonds (C-Cl, C-O vs C-I, C-Br), (3) Provide access to novel reactivity patterns through ligand-centered radicals.",
        "small_molecule": "N2 fixation, CO2 reduction, and H2 activation can be facilitated by redox-active ligands. The ligand provides multiple electrons needed for multi-electron substrate transformations that single metals cannot perform efficiently."
    },
    "characterization": {
        "spectroscopy": "Key methods for characterizing redox-active ligands: (1) EPR/ESR - detects paramagnetic ligand radical species, (2) UV-Vis NIR - intervalence charge transfer bands, (3) X-ray crystallography - bond length changes indicate oxidation state, (4) Cyclic voltammetry - reversible redox waves.",
        "determination": "To determine if a ligand is redox-active: (1) Compare metal oxidation states from spectroscopy vs. charge balance, (2) Look for discrepancies indicating ligand oxidation/reduction, (3) Structural evidence (bond lengths consistent with reduced/oxidized ligand), (4) Redox potentials matching ligand-based processes."
    }
}

def get_redox_context(query: str) -> str:
    """Get relevant context from the redox-active ligand knowledge base"""
    query_lower = query.lower()
    context_parts = []
    
    # Check for fundamental concepts
    if any(term in query_lower for term in ['what is', 'definition', 'innocent', 'non-innocent']):
        context_parts.append(f"Definition: {REDOX_KNOWLEDGE['fundamentals']['definition']}")
        if 'innocent' in query_lower or 'non-innocent' in query_lower:
            context_parts.append(f"\nInnocent vs Non-Innocent: {REDOX_KNOWLEDGE['fundamentals']['innocent_vs_noninnocent']}")
    
    # Check for specific ligand types
    ligand_keywords = {
        'pdi': ['pdi', 'bis(imino)pyridine', 'bis-imino', 'pyridine diimine'],
        'catecholate': ['catecholate', 'catechol', 'quinone', 'o-quinone', 'semiquinone'],
        'dithiolene': ['dithiolene', 'dithiolate'],
        'alpha_diimine': ['diimine', 'bipyridine', 'bipy', 'phenanthroline', 'phen'],
        'redox_nhc': ['nhc', 'carbene', 'nheterocyclic carbene']
    }
    
    for ligand_key, keywords in ligand_keywords.items():
        if any(kw in query_lower for kw in keywords):
            ligand_info = REDOX_KNOWLEDGE['ligand_classes'][ligand_key]
            context_parts.append(f"\nLigand: {ligand_info['name']}\n{ligand_info['description']}")
            if 'application' in query_lower or 'use' in query_lower:
                context_parts.append(f"Applications: {', '.join(ligand_info['applications'])}")
            break
    
    # Check for mechanism-related queries
    if any(term in query_lower for term in ['mechanism', 'how', 'electron reservoir', 'two-electron']):
        if 'electron reservoir' in query_lower or 'reservoir' in query_lower:
            context_parts.append(f"\nElectron Reservoir: {REDOX_KNOWLEDGE['mechanisms']['electron_reservoir']}")
        if 'two-electron' in query_lower:
            context_parts.append(f"\nTwo-Electron Processes: {REDOX_KNOWLEDGE['mechanisms']['two_electron_processes']}")
        if 'substrate' in query_lower or 'activation' in query_lower:
            context_parts.append(f"\nSubstrate Activation: {REDOX_KNOWLEDGE['mechanisms']['substrate_activation']}")
    
    # Check for application-related queries
    if any(term in query_lower for term in ['nickel', 'ni-catalyzed', 'ni catalyst']):
        context_parts.append(f"\nNickel Catalysis: {REDOX_KNOWLEDGE['applications']['nickel_catalysis']}")
    if 'c-o' in query_lower or 'co activation' in query_lower or 'c-o activation' in query_lower:
        context_parts.append(f"\nC-O Activation: {REDOX_KNOWLEDGE['applications']['c_o_activation']}")
    if 'cross-coupling' in query_lower or 'cross coupling' in query_lower or 'coupling' in query_lower:
        context_parts.append(f"\nCross-Coupling: {REDOX_KNOWLEDGE['applications']['cross_coupling']}")
    if any(term in query_lower for term in ['small molecule', 'n2', 'co2', 'h2', 'nitrogen', 'hydrogen']):
        context_parts.append(f"\nSmall Molecule Activation: {REDOX_KNOWLEDGE['applications']['small_molecule']}")
    
    # Check for characterization queries
    if any(term in query_lower for term in ['spectroscopy', 'characterization', 'epr', 'uv-vis', 'x-ray', 'crystallography', 'voltammetry']):
        context_parts.append(f"\nSpectroscopy: {REDOX_KNOWLEDGE['characterization']['spectroscopy']}")
    if any(term in query_lower for term in ['determine', 'how to', 'identify', 'evidence']):
        context_parts.append(f"\nDetermination: {REDOX_KNOWLEDGE['characterization']['determination']}")
    
    return '\n'.join(context_parts)


@app.route('/api/redox/chat', methods=['POST'])
@limiter.limit("20 per minute")
def redox_chat():
    """Chat endpoint for redox-active ligand assistant"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        provider = data.get('provider', None)  # User-specified provider
        api_key = data.get('api_key', None)    # User-specified API key
        model = data.get('model', None)        # User-specified model
        
        if not user_message.strip():
            return jsonify({'success': False, 'error': 'Empty message'}), 400
        
        # Get relevant context from knowledge base
        context = get_redox_context(user_message)
        
        # System prompt for redox-active ligand assistant
        system_prompt = """You are an expert assistant specializing in redox-active ligands and their applications in transition metal catalysis. Your expertise includes:

1. **Ligand Types**: Bis(imino)pyridines (PDI), catecholates/o-quinones, dithiolenes, α-diimines, redox-active NHCs, and other non-innocent ligands.

2. **Fundamental Concepts**: 
   - Distinction between innocent and non-innocent ligands
   - Metal-ligand cooperativity
   - Electron reservoir function
   - Two-electron processes with first-row metals

3. **Applications**:
   - Nickel-catalyzed C-O bond activation
   - Cross-coupling reactions (Suzuki, Heck, etc.)
   - Small molecule activation (N2, CO2, H2)
   - Biomass conversion

4. **Characterization Methods**:
   - EPR/ESR spectroscopy
   - UV-Vis NIR spectroscopy
   - X-ray crystallography
   - Cyclic voltammetry

When answering:
- Be accurate and scientifically rigorous
- Provide specific examples when relevant
- Explain mechanisms clearly
- Connect concepts to practical applications
- Mention key references or researchers when appropriate (e.g., Chirik, Wieghardt, Mindiola)
- Use appropriate chemical terminology

If a question is outside the scope of redox-active ligands or transition metal catalysis, politely redirect to relevant chemistry topics."""
        
        # Add context if available
        if context:
            system_prompt += f"\n\nRelevant context from knowledge base:\n{context}"
        
        # Get response from LLM - use user-provided credentials if available
        response = None
        llm_used = False
        error_msg = None
        
        if api_key and provider:
            # Use user-provided API key and provider
            try:
                from llm_providers import LLMProviderFactory
                print(f"[Redox Chat] Creating {provider} provider with user API key...")
                llm = LLMProviderFactory.create(provider, api_key=api_key)
                if model:
                    llm.model = model  # Override model if specified
                print(f"[Redox Chat] Calling {provider} with model {llm.model}...")
                response = llm.chat(system_prompt, user_message)
                if response:
                    llm_used = True
                    print(f"[Redox Chat] SUCCESS with user-provided {provider}")
                else:
                    error_msg = f"No response from {provider}"
                    print(f"[Redox Chat] No response from {provider}")
            except Exception as e:
                error_msg = str(e)
                print(f"[Redox Chat] Error with user provider: {e}")
                import traceback
                traceback.print_exc()
                # Fall through to default provider
        
        if not response:
            # Try default provider (environment variables)
            response = get_llm_response_sync(system_prompt, user_message)
            if response:
                llm_used = True
        
        if response:
            # Generate relevant references
            references = []
            if 'pdi' in user_message.lower() or 'bis(imino)pyridine' in user_message.lower():
                references.append({
                    'title': 'Chirik, P. J. - PDI Chemistry Reviews',
                    'url': 'https://pubs.acs.org/journals/orgnd7'
                })
            if 'catecholate' in user_message.lower() or 'quinone' in user_message.lower():
                references.append({
                    'title': 'Wieghardt, K. - Redox-Active Ligands Reviews',
                    'url': 'https://pubs.acs.org/journals/inocaj'
                })
            if 'nickel' in user_message.lower() and 'c-o' in user_message.lower():
                references.append({
                    'title': 'Ni-Catalyzed C-O Activation',
                    'url': 'https://pubs.acs.org/journals/orgnd7'
                })
            
            return jsonify({
                'success': True,
                'response': response,
                'references': references,
                'llm_used': llm_used,
                'provider': provider if api_key else 'default'
            })
        else:
            # Fallback response if LLM unavailable
            fallback_msg = get_redox_fallback_response(user_message)
            if error_msg:
                fallback_msg = f"**AI Error:** {error_msg}\n\n{fallback_msg}"
            return jsonify({
                'success': True,
                'response': fallback_msg,
                'references': [],
                'llm_used': False,
                'offline_mode': True,
                'error': error_msg
            })
            
    except Exception as e:
        logger.error(f"Redox chat error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'An internal error occurred'}), 500


def get_redox_fallback_response(query: str) -> str:
    """Provide fallback response when LLM is unavailable"""
    query_lower = query.lower()
    
    # Direct matches from knowledge base
    if 'what are redox' in query_lower or 'definition' in query_lower:
        return REDOX_KNOWLEDGE['fundamentals']['definition']
    
    if 'innocent' in query_lower:
        return REDOX_KNOWLEDGE['fundamentals']['innocent_vs_noninnocent']
    
    if 'pdi' in query_lower or 'bis(imino)pyridine' in query_lower:
        ligand = REDOX_KNOWLEDGE['ligand_classes']['pdi']
        return f"{ligand['name']}: {ligand['description']} Applications include: {', '.join(ligand['applications'])}."
    
    if 'catecholate' in query_lower or 'quinone' in query_lower:
        ligand = REDOX_KNOWLEDGE['ligand_classes']['catecholate']
        return f"{ligand['name']}: {ligand['description']} Redox states: {', '.join(ligand['redox_states'])}."
    
    if 'electron reservoir' in query_lower:
        return REDOX_KNOWLEDGE['mechanisms']['electron_reservoir']
    
    if 'nickel' in query_lower:
        return REDOX_KNOWLEDGE['applications']['nickel_catalysis']
    
    if 'c-o' in query_lower:
        return REDOX_KNOWLEDGE['applications']['c_o_activation']
    
    # Generic response
    return """I'm currently operating in offline mode with limited responses. For detailed questions about redox-active ligands, please ensure the AI service is connected. 

In the meantime, here are some key topics I can discuss:
- Bis(imino)pyridine (PDI) ligands and their applications
- Catecholate/o-quinone redox chemistry
- Nickel-catalyzed C-O bond activation
- Metal-ligand cooperativity
- Characterization methods for redox-active ligands

Please try asking about one of these topics!"""


# ==================== NIOCOBOT API (Nickel Catalysis Assistant) ====================

import json
import requests

# Load NiCOBot data files
NIOCOBOT_DATA_DIR = os.path.join(BACKEND_DIR, 'nicobot_data')

def load_nicobot_data():
    """Load NiCOBot electrophile and nucleophile data"""
    data = {
        'electrophiles': {},
        'nucleophiles': {}
    }
    
    try:
        # Load electrophile data
        e_lvg_path = os.path.join(NIOCOBOT_DATA_DIR, 'E_LVG_name_smiles.json')
        if os.path.exists(e_lvg_path):
            with open(e_lvg_path, 'r') as f:
                data['electrophiles'] = json.load(f)
        
        # Load nucleophile data
        nu_lvg_path = os.path.join(NIOCOBOT_DATA_DIR, 'Nu_LVG_name_smiles.json')
        if os.path.exists(nu_lvg_path):
            with open(nu_lvg_path, 'r') as f:
                data['nucleophiles'] = json.load(f)
                
    except Exception as e:
        print(f"[NiCOBot] Error loading data: {e}")
    
    return data

NIOCOBOT_DATA = load_nicobot_data()

# NiCOBot specialized knowledge base
NIOCOBOT_KNOWLEDGE = {
    "catalysis": {
        "nickel_overview": "Nickel catalysis offers a cost-effective alternative to palladium for cross-coupling reactions. Nickel can activate strong bonds (C-Cl, C-O) that palladium cannot, making it valuable for sustainable chemistry using biomass-derived feedstocks.",
        "advantages": "Nickel is approximately 1000x cheaper than palladium, can activate inert C-Cl and C-O bonds, supports multiple oxidation states (Ni(0), Ni(I), Ni(II), Ni(III)), and enables novel reactivity patterns.",
        "mechanisms": "Ni-catalyzed reactions typically proceed through: (1) Oxidative addition of C-X bond to Ni(0), (2) Transmetalation with nucleophile, (3) Reductive elimination to form product. Single-electron pathways are also possible via Ni(I)/Ni(III) intermediates."
    },
    "c_o_activation": {
        "overview": "C-O bond activation in aryl ethers and esters is challenging due to strong bonds (~85-90 kcal/mol). Nickel catalysts with appropriate ligands can cleave these bonds, enabling conversion of biomass-derived compounds.",
        "leaving_groups": "Common C-O leaving groups include: Triflates (weak bond, most reactive), Tosylates (weak bond), Mesylates (weak bond), Acetates (medium bond), Pivalates (medium bond), Aryl ethers (inert bond, most challenging)",
        "conditions": "Typical conditions: Ni(cod)2 or NiCl2 as precatalyst, phosphine or NHC ligands, base (Cs2CO3, K3PO4), 80-120°C, 12-24 hours."
    },
    "cross_coupling": {
        "suzuki": "Suzuki-Miyaura coupling: Ni-catalyzed C-C bond formation between organoboron reagents and organic electrophiles. Advantage: boron reagents are non-toxic and stable.",
        "kumada": "Kumada coupling: Uses Grignard reagents as nucleophiles. Very reactive but requires strict anhydrous conditions.",
        "negishi": "Negishi coupling: Uses organozinc reagents. More functional group tolerant than Kumada.",
        "stille": "Stille coupling: Uses organotin reagents. Wide substrate scope but toxicity concerns."
    },
    "electrophiles": {
        "definition": "Electrophiles are electron-deficient species that accept electrons in reactions. In Ni-catalyzed C-O activation, electrophiles are typically aryl esters, ethers, or halides.",
        "types": ["Triflates", "Tosylates", "Mesylates", "Acetates", "Pivalates", "Aryl methyl ethers", "Aryl halides"]
    },
    "nucleophiles": {
        "definition": "Nucleophiles are electron-rich species that donate electrons. Common nucleophiles in Ni catalysis include organoboron, organomagnesium (Grignard), and organozinc reagents.",
        "types": ["Boronic acids/esters", "Grignard reagents (R-MgX)", "Organozinc reagents (R-ZnX)", "Organosilanes"]
    }
}

def get_nicobot_context(query: str) -> str:
    """Get relevant context from NiCOBot knowledge base"""
    context_parts = []
    query_lower = query.lower()
    
    # Check for relevant topics
    if 'nickel' in query_lower or 'ni-catalyzed' in query_lower:
        context_parts.append(f"Nickel Catalysis: {NIOCOBOT_KNOWLEDGE['catalysis']['nickel_overview']}")
    
    if 'c-o' in query_lower or 'bond activation' in query_lower or 'ether' in query_lower or 'ester' in query_lower:
        context_parts.append(f"\nC-O Activation: {NIOCOBOT_KNOWLEDGE['c_o_activation']['overview']}")
        context_parts.append(f"\nLeaving Groups: {NIOCOBOT_KNOWLEDGE['c_o_activation']['leaving_groups']}")
    
    if 'suzuki' in query_lower:
        context_parts.append(f"\nSuzuki Coupling: {NIOCOBOT_KNOWLEDGE['cross_coupling']['suzuki']}")
    
    if 'kumada' in query_lower:
        context_parts.append(f"\nKumada Coupling: {NIOCOBOT_KNOWLEDGE['cross_coupling']['kumada']}")
    
    if 'negishi' in query_lower:
        context_parts.append(f"\nNegishi Coupling: {NIOCOBOT_KNOWLEDGE['cross_coupling']['negishi']}")
    
    if 'electrophile' in query_lower:
        context_parts.append(f"\nElectrophiles: {NIOCOBOT_KNOWLEDGE['electrophiles']['definition']}")
        context_parts.append(f"Types: {', '.join(NIOCOBOT_KNOWLEDGE['electrophiles']['types'])}")
    
    if 'nucleophile' in query_lower:
        context_parts.append(f"\nNucleophiles: {NIOCOBOT_KNOWLEDGE['nucleophiles']['definition']}")
        context_parts.append(f"Types: {', '.join(NIOCOBOT_KNOWLEDGE['nucleophiles']['types'])}")
    
    return '\n'.join(context_parts)

def query2smiles_nicobot(query: str) -> str:
    """Convert molecule name to SMILES using PubChem API"""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/property/IsomericSMILES/JSON"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
        return None
    except Exception as e:
        print(f"[NiCOBot] PubChem query error: {e}")
        return None

def check_e_or_nu(smiles: str) -> str:
    """Check if a molecule is an electrophile or nucleophile based on SMILES"""
    if 'Mg' in smiles or 'B' in smiles or 'Zn' in smiles:
        return "nucleophile"
    return "electrophile"

def check_co_bond_strength(smiles: str) -> str:
    """Estimate C-O bond strength based on functional groups"""
    # Check for weak C-O bonds (triflates, tosylates, mesylates)
    weak_patterns = ['S(=O)(=O)O', 'OS(=O)', 'S(=O)(=O)']
    medium_patterns = ['OC(=O)', 'C(=O)O', 'C(=O)OC']
    
    for pattern in weak_patterns:
        if pattern in smiles:
            return "weak (triflate/tosylate/mesylate - very reactive)"
    
    for pattern in medium_patterns:
        if pattern in smiles:
            return "medium (ester/acetate - moderate reactivity)"
    
    return "inert (aryl ether - challenging substrate)"

def find_similar_electrophile(smiles: str) -> list:
    """Find similar electrophiles from the database"""
    similar = []
    for e_smiles, names in list(NIOCOBOT_DATA['electrophiles'].items())[:20]:
        if len(similar) >= 5:
            break
        if names:
            similar.append({
                'smiles': e_smiles,
                'name': names[0] if isinstance(names, list) else names
            })
    return similar

def find_similar_nucleophile(smiles: str) -> list:
    """Find similar nucleophiles from the database"""
    similar = []
    for n_smiles, names in list(NIOCOBOT_DATA['nucleophiles'].items())[:20]:
        if len(similar) >= 5:
            break
        if names:
            similar.append({
                'smiles': n_smiles,
                'name': names[0] if isinstance(names, list) else names
            })
    return similar

@app.route('/api/nicobot/chat', methods=['POST'])
@limiter.limit("20 per minute")
def nicobot_chat():
    """Chat endpoint for NiCOBot assistant"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        provider = data.get('provider', None)
        api_key = data.get('api_key', None)
        model = data.get('model', None)
        
        if not user_message.strip():
            return jsonify({'success': False, 'error': 'Empty message'}), 400
        
        # Get relevant context
        context = get_nicobot_context(user_message)
        
        # System prompt for NiCOBot
        system_prompt = """You are NiCOBot, an expert AI assistant specializing in Nickel-catalyzed cross-coupling reactions and C-O bond activation chemistry. Your expertise includes:

1. **Nickel Catalysis Fundamentals**:
   - Oxidative addition, transmetalation, reductive elimination mechanisms
   - Ni(0)/Ni(II) and Ni(I)/Ni(III) catalytic cycles
   - Advantages over palladium catalysis (cost, C-Cl/C-O activation)

2. **C-O Bond Activation**:
   - Leaving group reactivity: triflates > tosylates > mesylates > acetates > pivalates > aryl ethers
   - Bond dissociation energies and activation strategies
   - Biomass conversion applications

3. **Cross-Coupling Reactions**:
   - Suzuki-Miyaura (organoboron reagents)
   - Kumada (Grignard reagents)
   - Negishi (organozinc reagents)
   - Stille (organotin reagents)

4. **Reaction Components**:
   - Electrophiles: aryl halides, triflates, tosylates, mesylates, esters, ethers
   - Nucleophiles: boronic acids, Grignard reagents, organozinc compounds
   - Ligands: phosphines (PCy3, PPh3), NHCs, bipyridines
   - Bases: Cs2CO3, K3PO4, NaOtBu

5. **Practical Guidance**:
   - Catalyst selection and loading
   - Solvent and temperature optimization
   - Functional group compatibility
   - Troubleshooting failed reactions

When answering:
- Provide specific reaction conditions when applicable
- Explain mechanistic rationale
- Suggest optimal reagents and conditions
- Cite relevant literature or researchers when appropriate (e.g., Weix, Jamison, Martin)
- Be practical and actionable in your recommendations"""

        if context:
            system_prompt += f"\n\nRelevant context:\n{context}"
        
        # Get response from LLM
        response = None
        llm_used = False
        error_msg = None
        
        if api_key and provider:
            try:
                from llm_providers import LLMProviderFactory
                print(f"[NiCOBot] Creating {provider} provider with user API key...")
                llm = LLMProviderFactory.create(provider, api_key=api_key)
                if model:
                    llm.model = model
                response = llm.chat(system_prompt, user_message)
                if response:
                    llm_used = True
            except Exception as e:
                error_msg = str(e)
                print(f"[NiCOBot] Error: {e}")
        
        if not response:
            response = get_llm_response_sync(system_prompt, user_message)
            if response:
                llm_used = True
        
        if response:
            return jsonify({
                'success': True,
                'response': response,
                'llm_used': llm_used
            })
        else:
            return jsonify({
                'success': True,
                'response': get_nicobot_fallback_response(user_message),
                'llm_used': False,
                'offline_mode': True,
                'error': error_msg
            })
            
    except Exception as e:
        logger.error(f"NiCOBot chat error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'An internal error occurred'}), 500

# Alias for TMC chat
@app.route('/api/chat', methods=['POST'])
@limiter.limit("20 per minute")
def tmc_chat():
    """Alias for NiCOBot chat - used by TMC page"""
    return nicobot_chat()

@app.route('/api/nicobot/smiles/<compound_name>')
def nicobot_get_smiles(compound_name):
    """Convert compound name to SMILES"""
    smiles = query2smiles_nicobot(compound_name)
    if smiles:
        e_or_nu = check_e_or_nu(smiles)
        co_strength = check_co_bond_strength(smiles)
        return jsonify({
            'success': True,
            'name': compound_name,
            'smiles': smiles,
            'type': e_or_nu,
            'co_bond_strength': co_strength
        })
    return jsonify({'success': False, 'error': 'Compound not found'}), 404

@app.route('/api/nicobot/check', methods=['POST'])
def nicobot_check_molecule():
    """Check if a molecule is electrophile/nucleophile and C-O bond strength"""
    try:
        data = request.get_json()
        smiles = data.get('smiles', '')
        
        if not smiles:
            return jsonify({'success': False, 'error': 'No SMILES provided'}), 400
        
        result = {
            'smiles': smiles,
            'type': check_e_or_nu(smiles),
            'co_bond_strength': check_co_bond_strength(smiles)
        }
        
        # Find similar compounds
        if result['type'] == 'electrophile':
            result['similar_compounds'] = find_similar_electrophile(smiles)
        else:
            result['similar_compounds'] = find_similar_nucleophile(smiles)
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def get_nicobot_fallback_response(query: str) -> str:
    """Provide fallback response when LLM is unavailable"""
    query_lower = query.lower()
    
    if 'c-o' in query_lower or 'bond activation' in query_lower:
        return NIOCOBOT_KNOWLEDGE['c_o_activation']['overview']
    
    if 'nickel' in query_lower:
        return NIOCOBOT_KNOWLEDGE['catalysis']['nickel_overview']
    
    if 'suzuki' in query_lower:
        return NIOCOBOT_KNOWLEDGE['cross_coupling']['suzuki']
    
    if 'electrophile' in query_lower:
        return f"{NIOCOBOT_KNOWLEDGE['electrophiles']['definition']} Types: {', '.join(NIOCOBOT_KNOWLEDGE['electrophiles']['types'])}"
    
    if 'nucleophile' in query_lower:
        return f"{NIOCOBOT_KNOWLEDGE['nucleophiles']['definition']} Types: {', '.join(NIOCOBOT_KNOWLEDGE['nucleophiles']['types'])}"
    
    return """I'm currently in offline mode. I can help with:

**Ni-Catalyzed Reactions:**
- C-O bond activation (ethers, esters)
- Cross-coupling (Suzuki, Kumada, Negishi)
- Catalyst and ligand selection

**Quick Topics:**
- C-O bond activation
- Nickel catalysis advantages
- Electrophiles vs Nucleophiles
- Suzuki coupling
- Reaction conditions

Please configure your API key in settings for full AI responses!"""


# ==================== RUN ====================

if __name__ == '__main__':
    # Configuration from environment
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    PORT = int(os.environ.get('PORT', 5000))
    HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
    
    print("\n" + "="*50)
    print("GenAI Research Flask Server")
    print("="*50)
    print(f"Static files served from: {STATIC_FOLDER}")
    print(f"Access the site at: http://{HOST}:{PORT}")
    print(f"Debug mode: {DEBUG}")
    print("="*50 + "\n")
    
    app.run(debug=DEBUG, host=HOST, port=PORT)
