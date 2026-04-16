# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import networkx as nx

# Import your logic modules
from modules.Forward_Reference import build_forward_network
from modules.Local_Reference import build_reference_network
from modules.Cross_Reference import build_cross_reference_network

app = Flask(__name__)
# Enable CORS to allow your HTML file to communicate with this server
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def log_progress(msg):
    """Helper to replace Streamlit progress with console logs"""
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
        all_papers = [] # This will hold the data for the Excel download
        
        try:
            # 1. Build Graph based on type
            if analysis_type == "forward":
                # Unpack 3 values: (Filtered Graph for Viz, Suggestions, Full Data List)
                G, suggestions, all_papers = build_forward_network(filepath, progress_callback=log_progress)
                
            elif analysis_type == "backward":
                # Unpack 3 values: (Filtered Graph for Viz, Suggestions, Full Data List)
                G, suggestions, all_papers = build_reference_network(filepath, progress_callback=log_progress)
                
            elif analysis_type == "cross":
                # Cross reference returns the full graph G directly
                G = build_cross_reference_network(filepath, progress_callback=log_progress)
                
                # Manually build all_papers list for Cross Reference since the function wasn't modified to return it
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
                return jsonify({'error': 'Could not build network (too few nodes or API error). Check console.'}), 400

            # 2. Convert NetworkX Graph to Cytoscape.js JSON format
            # We use edges="edges" to ensure compatibility with Cytoscape
            graph_json = nx.node_link_data(G, edges="edges")

            return jsonify({
                'elements': graph_json, 
                'suggestions': suggestions,
                'all_papers': all_papers, # Send full data for Excel download
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
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    print("Starting Python Backend Server...")
    print("Access the Citation Tool via your HTML page.")
    app.run(debug=True, port=5000)