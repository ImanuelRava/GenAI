# app.py - ChemAI Research Flask Application
# Complete Flask application for PythonAnywhere deployment

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import networkx as nx

# Import your logic modules
from modules.Forward_Reference import build_forward_network
from modules.Local_Reference import build_reference_network
from modules.Cross_Reference import build_cross_reference_network

# Create Flask app with template and static folders
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# Enable CORS
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==================== AI CHAT ====================
# System prompt for chemistry assistant
SYSTEM_PROMPT = """You are ChemAI, an expert chemistry assistant specializing in:
- Nickel-catalyzed cross-coupling reactions
- C-O bond activation mechanisms
- Transition metal catalysis
- Organic synthesis methodology
- Scientific research methodology

You help students and researchers understand complex chemistry concepts. Be clear, accurate, and educational in your responses.
When explaining mechanisms, use simple language and analogies where appropriate.
Always cite relevant literature or concepts when discussing specific reactions."""

def get_ai_response(message, history=None):
    """Get AI response using the AI SDK"""
    try:
        # Import the AI SDK
        import json
        import urllib.request
        
        # Build messages array
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        if history:
            for msg in history[-10:]:  # Last 10 messages for context
                role = "user" if msg.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": msg.get("content", "")})
        
        messages.append({"role": "user", "content": message})
        
        # Use the AI SDK or fallback to a simple response
        try:
            # Try to use the z-ai-web-dev-sdk if available
            from z_ai_web_dev_sdk import ZAI
            
            async def get_response():
                zai = await ZAI.create()
                completion = await zai.chat.completions.create(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                return completion.choices[0].message.content
            
            import asyncio
            response = asyncio.run(get_response())
            return response
            
        except ImportError:
            # Fallback: Use requests to call the API directly
            import requests
            
            # This is a placeholder - in production, use the actual AI API
            # For now, return a helpful message about the topic
            return generate_fallback_response(message)
            
    except Exception as e:
        print(f"AI Error: {e}")
        return f"I apologize, but I'm having trouble connecting to my AI backend. Error: {str(e)}"

def generate_fallback_response(message):
    """Generate a fallback response based on keywords"""
    message_lower = message.lower()
    
    if 'cross-coupling' in message_lower or 'cross coupling' in message_lower:
        return """Cross-coupling reactions are powerful synthetic methods that join two organic molecules using a metal catalyst. 

Key points:
1. **Mechanism**: Involves oxidative addition, transmetalation, and reductive elimination
2. **Catalysts**: Commonly Pd, Ni, Cu, or other transition metals
3. **Applications**: Pharmaceutical synthesis, materials science, natural product synthesis

Would you like me to explain any specific aspect in more detail?"""

    elif 'oxidative addition' in message_lower:
        return """Oxidative addition is the first step in many catalytic cycles where a metal center inserts into a chemical bond.

**For cross-coupling:**
- The metal catalyst (M⁰) reacts with an organic halide (R-X)
- The metal oxidizes from M⁰ to M²⁺
- The R-X bond breaks, forming M-R and M-X bonds

**Example with Nickel:**
Ni⁰ + Ar-Br → Ni²⁺(Ar)(Br)

This step is often rate-determining and depends on the substrate's reactivity (I > Br > Cl > F)."""

    elif 'nickel' in message_lower or 'ni-catalyzed' in message_lower:
        return """Nickel catalysts are excellent for C-O bond activation due to:

1. **Cost-effective**: Much cheaper than palladium
2. **Reactivity**: Can activate strong C-O bonds that Pd cannot
3. **Versatility**: Works with various electrophiles including phenols, esters, and ethers

**Key advantages for C-O activation:**
- Lower electronegativity allows oxidative addition into strong C-O bonds
- Can use biomass-derived substrates (lignin model compounds)
- Enables sustainable chemistry approaches

Would you like to know about specific Ni-catalyzed transformations?"""

    elif 'suzuki' in message_lower:
        return """The Suzuki-Miyaura coupling is one of the most widely used cross-coupling reactions.

**General reaction:**
R-B(OH)₂ + R'-X → R-R' (catalyzed by Pd or Ni)

**Key features:**
- Uses boronic acids/boronic esters as nucleophiles
- Mild conditions, functional group tolerant
- Requires a base (often K₂CO₃, K₃PO₄)

**Advantages:**
- Air-stable, non-toxic boron reagents
- Works in aqueous solvents
- Commercially available boronic acids

Compare this with Negishi coupling which uses organozinc reagents!"""

    elif 'negishi' in message_lower:
        return """Negishi coupling uses organozinc reagents for C-C bond formation.

**General reaction:**
R-ZnX + R'-X → R-R' (Pd or Ni catalyzed)

**Key features:**
- Highly reactive organozinc nucleophiles
- Excellent functional group compatibility
- Works at low temperatures

**Comparison with Suzuki:**
- More reactive, works with less reactive electrophiles
- Broader scope for certain substrates
- Zinc reagents are air-sensitive (unlike boron)

This reaction won Ei-ichi Negishi the 2010 Nobel Prize!"""

    else:
        return f"""Thank you for your question about "{message}".

I'm your chemistry AI assistant, specializing in cross-coupling reactions and transition metal catalysis. I can help you understand:

- Cross-coupling mechanisms (oxidative addition, transmetalation, reductive elimination)
- Different coupling types (Suzuki, Negishi, Stille, Kumada, etc.)
- Nickel catalysis and C-O bond activation
- Catalyst design and ligand selection

Please ask me a specific question, and I'll provide a detailed explanation!"""

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/nicobot')
def nicobot_intro():
    """NiCOBot learning introduction page"""
    return render_template('nicobot_intro.html',
                          progress='2/14',
                          progress_percent=14)

@app.route('/nicobot/lecture/<int:lecture_id>')
def lecture(lecture_id):
    """Individual lecture pages"""
    # Lecture content database (in production, use a database)
    lectures = {
        1: {
            'id': 1,
            'title': 'The Basics of Cross-Coupling',
            'reading_time': '10 min read',
            'completion_time': '20 min',
            'key_points': [
                'Cross-coupling joins two molecules using a metal catalyst',
                'The reaction requires an electrophile (usually R-X) and a nucleophile (R-M)',
                'Common metals include Pd, Ni, Cu, and Fe',
                'Cross-coupling revolutionized organic synthesis - 2010 Nobel Prize',
            ],
            'suggested_prompts': [
                'What is cross-coupling?',
                'Why are metal catalysts needed?',
                'Compare Suzuki and Negishi coupling',
            ],
            'content': '''
                <h2>Introduction to Cross-Coupling</h2>
                <p>Cross-coupling reactions are among the most important transformations in modern organic chemistry. They allow chemists to form carbon-carbon bonds efficiently and selectively, enabling the synthesis of complex molecules that would be difficult or impossible to make by other means.</p>
                
                <h3>The Basic Concept</h3>
                <p>In a cross-coupling reaction, two organic fragments are joined together with the help of a transition metal catalyst. One fragment serves as the <strong>electrophile</strong> (typically an organic halide, R-X), and the other serves as the <strong>nucleophile</strong> (an organometallic reagent, R'-M).</p>
                
                <p>The general equation for cross-coupling is:</p>
                <div style="background: rgba(0,242,255,0.1); padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0;">
                    <strong>R-X + R'-M → R-R' + M-X</strong>
                    <br><span style="font-size: 12px; color: var(--text-slate-400);">Catalyzed by transition metal (Pd, Ni, etc.)</span>
                </div>
                
                <h3>Why Use Metal Catalysts?</h3>
                <p>Without catalysts, joining two carbon atoms directly requires harsh conditions and often leads to unwanted side products. Transition metals like palladium and nickel can:</p>
                <ul style="margin-left: 20px; color: var(--text-slate-400);">
                    <li>Facilitate bond breaking and formation under mild conditions</li>
                    <li>Provide excellent selectivity (chemo-, regio-, and stereoselectivity)</li>
                    <li>Enable reactions that would otherwise be impossible</li>
                </ul>
                
                <h3>Historical Significance</h3>
                <p>The development of cross-coupling reactions earned Richard Heck, Ei-ichi Negishi, and Akira Suzuki the 2010 Nobel Prize in Chemistry. These reactions have become indispensable in:</p>
                <ul style="margin-left: 20px; color: var(--text-slate-400);">
                    <li>Pharmaceutical manufacturing</li>
                    <li>Materials science (OLEDs, conductive polymers)</li>
                    <li>Natural product synthesis</li>
                    <li>Agrochemicals</li>
                </ul>
            ''',
            'prev_id': None,
            'next_id': 2,
        },
        2: {
            'id': 2,
            'title': 'The Role of Transition Metals',
            'reading_time': '12 min read',
            'completion_time': '25 min',
            'key_points': [
                'Transition metals have partially filled d-orbitals',
                'These metals can easily change oxidation states',
                'Pd(0)/Pd(II) and Ni(0)/Ni(II) are common catalytic cycles',
                'The metal acts as a "matchmaker" between two carbon fragments',
            ],
            'suggested_prompts': [
                'Why use transition metals as catalysts?',
                'What makes nickel special?',
                'Compare Pd and Ni catalysis',
            ],
            'content': '''
                <h2>Why Transition Metals?</h2>
                <p>Transition metals are the heart of cross-coupling catalysis. Their unique electronic properties make them perfect "molecular matchmakers" that can bring two carbon fragments together.</p>
                
                <h3>The d-Orbital Advantage</h3>
                <p>Transition metals have partially filled d-orbitals that can accept and donate electrons readily. This allows them to:</p>
                <ul style="margin-left: 20px; color: var(--text-slate-400);">
                    <li><strong>Coordinate</strong> to organic molecules</li>
                    <li><strong>Accept electrons</strong> (oxidative addition)</li>
                    <li><strong>Donate electrons</strong> (reductive elimination)</li>
                    <li><strong>Cycle between oxidation states</strong> catalytically</li>
                </ul>
                
                <h3>Common Catalytic Metals</h3>
                <p><strong>Palladium (Pd)</strong> - The gold standard:</p>
                <ul style="margin-left: 20px; color: var(--text-slate-400);">
                    <li>Excellent for C-C and C-heteroatom coupling</li>
                    <li>Well-understood mechanisms</li>
                    <li>High functional group tolerance</li>
                    <li>Expensive but highly efficient</li>
                </ul>
                
                <p><strong>Nickel (Ni)</strong> - The rising star:</p>
                <ul style="margin-left: 20px; color: var(--text-slate-400);">
                    <li>Activates strong bonds (C-Cl, C-O, C-F)</li>
                    <li>Much cheaper than palladium</li>
                    <li>Enables unique reactivity patterns</li>
                    <li>Key for sustainable chemistry</li>
                </ul>
                
                <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(20, 184, 166, 0.1)); border: 1px solid rgba(16, 185, 129, 0.3); padding: 20px; border-radius: 12px; margin: 20px 0;">
                    <h4 style="color: #34d399; margin-bottom: 10px;">💡 Key Insight</h4>
                    <p style="color: var(--text-slate-400); margin: 0;">
                        Nickel's lower electronegativity compared to palladium allows it to insert into stronger bonds like C-O bonds. This is why Ni is the catalyst of choice for activating biomass-derived compounds!
                    </p>
                </div>
            ''',
            'prev_id': 1,
            'next_id': 3,
        },
        3: {
            'id': 3,
            'title': 'The Catalytic Cycle',
            'reading_time': '15 min read',
            'completion_time': '30 min',
            'key_points': [
                'Three key steps: Oxidative Addition → Transmetalation → Reductive Elimination',
                'Each step has specific requirements and can be optimized',
                'Ligands play a crucial role in controlling reactivity',
                'Understanding the cycle helps troubleshoot reactions',
            ],
            'suggested_prompts': [
                'Explain oxidative addition',
                'What is transmetalation?',
                'How does reductive elimination work?',
            ],
            'content': '''
                <h2>The Three Steps of Cross-Coupling</h2>
                <p>Every cross-coupling reaction follows the same fundamental catalytic cycle. Understanding this cycle is key to designing and optimizing reactions.</p>
                
                <h3>Step 1: Oxidative Addition</h3>
                <p>The metal catalyst (in its zero-valent state) inserts into the electrophile's bond, typically the carbon-halogen bond.</p>
                <div style="background: rgba(0,242,255,0.1); padding: 15px; border-radius: 8px; text-align: center; margin: 15px 0;">
                    <strong>M⁰ + R-X → M²⁺(R)(X)</strong>
                </div>
                <p style="color: var(--text-slate-400);">The metal is oxidized from M⁰ to M²⁺, and both R and X become ligands on the metal.</p>
                
                <h3>Step 2: Transmetalation</h3>
                <p>The nucleophile (organometallic reagent) transfers its organic group to the metal, displacing X.</p>
                <div style="background: rgba(112,0,255,0.1); padding: 15px; border-radius: 8px; text-align: center; margin: 15px 0;">
                    <strong>M²⁺(R)(X) + R'-M' → M²⁺(R)(R') + M'-X</strong>
                </div>
                <p style="color: var(--text-slate-400);">Now both organic groups are bound to the same metal center!</p>
                
                <h3>Step 3: Reductive Elimination</h3>
                <p>The two organic groups couple and leave the metal, regenerating the catalyst.</p>
                <div style="background: rgba(0,255,136,0.1); padding: 15px; border-radius: 8px; text-align: center; margin: 15px 0;">
                    <strong>M²⁺(R)(R') → M⁰ + R-R'</strong>
                </div>
                <p style="color: var(--text-slate-400);">The catalyst returns to M⁰, ready for another cycle!</p>
                
                <div style="background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(236, 72, 153, 0.1)); border: 1px solid rgba(168, 85, 247, 0.2); padding: 20px; border-radius: 12px; margin: 20px 0;">
                    <h4 style="color: #c084fc; margin-bottom: 10px;">🔄 The Cycle Continues</h4>
                    <p style="color: var(--text-slate-400); margin: 0;">
                        Since the catalyst is regenerated after each cycle, only a small amount (typically 1-5 mol%) is needed. This is what makes cross-coupling so efficient and economical!
                    </p>
                </div>
            ''',
            'prev_id': 2,
            'next_id': None,
        }
    }
    
    lecture = lectures.get(lecture_id)
    if not lecture:
        return render_template('index.html'), 404
    
    return render_template('lecture.html', 
                          lecture=lecture,
                          current_lesson=lecture_id)

@app.route('/redox-ligands')
def redox_ligands():
    """Redox active ligands page"""
    return render_template('index.html')

@app.route('/citation-tool')
def citation_tool():
    """Citation network tool page"""
    return send_from_directory('.', 'citation-tool.html')

@app.route('/quiz')
def quiz():
    """Chemistry quiz page"""
    return send_from_directory('.', 'quiz.html')

@app.route('/data-extraction')
def data_extraction():
    """Data extraction tool page"""
    return send_from_directory('.', 'data-extraction.html')

@app.route('/nicobot-live')
def nicobot_live():
    """NiCOBot live prediction page"""
    return send_from_directory('.', 'nicobot-embed.html')

# ==================== API ROUTES ====================

@app.route('/api/network', methods=['POST'])
def analyze_network():
    """Analyze citation network"""
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
                G, suggestions, all_papers = build_forward_network(filepath, progress_callback=lambda x: print(x))
                
            elif analysis_type == "backward":
                G, suggestions, all_papers = build_reference_network(filepath, progress_callback=lambda x: print(x))
                
            elif analysis_type == "cross":
                G = build_cross_reference_network(filepath, progress_callback=lambda x: print(x))
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
                return jsonify({'error': 'Could not build network (too few nodes or API error)'}), 400

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

@app.route('/api/chat', methods=['POST'])
def chat():
    """AI Chat endpoint"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        history = data.get('history', [])
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        response = get_ai_response(message, history)
        
        return jsonify({'response': response})
    
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({'error': str(e)}), 500

# ==================== STATIC FILES ====================

@app.route('/css/<path:filename>')
def serve_css(filename):
    """Serve CSS files from static folder"""
    return send_from_directory('static/css', filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    """Serve JS files from static folder"""
    return send_from_directory('static/js', filename)

@app.route('/images/<path:filename>')
def serve_images(filename):
    """Serve image files"""
    return send_from_directory('static/images', filename)

# ==================== MAIN ====================

if __name__ == '__main__':
    print("=" * 50)
    print("ChemAI Research - Flask Server")
    print("=" * 50)
    print("Access the application at: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
