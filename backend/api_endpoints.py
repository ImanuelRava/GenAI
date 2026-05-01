# ==================== UPDATED LLM KNOWLEDGE GRAPH API ====================
# Add these updated endpoints to your app.py to support dynamic API key configuration

import os
import json
from flask import request, jsonify
from llm_providers import get_llm_response, generate_knowledge_graph as llm_generate_kg, explain_concept


def get_llm_response_with_key(system_prompt: str, user_message: str, api_key: str = None, provider: str = None) -> str:
    """
    Get response from LLM with optional dynamic API key.
    
    Priority:
    1. Use provided api_key if available
    2. Fall back to environment variables
    """
    # Temporarily set API key if provided
    original_key = None
    key_name = None
    
    if api_key:
        # Map provider to environment variable name
        provider_key_map = {
            'deepseek': 'DEEPSEEK_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'groq': 'GROQ_API_KEY',
            'gemini': 'GEMINI_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY',
            'huggingface': 'HF_API_KEY',
            'hf': 'HF_API_KEY'
        }
        
        provider_lower = (provider or 'deepseek').lower()
        key_name = provider_key_map.get(provider_lower, 'DEEPSEEK_API_KEY')
        
        # Store original and set new key
        original_key = os.environ.get(key_name)
        os.environ[key_name] = api_key
    
    try:
        # Call LLM with specified provider
        actual_provider = provider if provider else None
        response = get_llm_response(system_prompt, user_message, provider=actual_provider)
        return response
    finally:
        # Restore original environment variable
        if api_key and key_name:
            if original_key is not None:
                os.environ[key_name] = original_key
            else:
                os.environ.pop(key_name, None)


def generate_knowledge_graph_with_llm(topic: str, api_key: str = None, provider: str = None) -> dict:
    """
    Use LLM to generate a knowledge graph for a given topic.
    Supports dynamic API key configuration.
    """
    system_prompt = """You are an expert in transition metal catalysis and chemistry education.
Generate a knowledge graph for the given topic in valid JSON format ONLY (no markdown, no explanation).
Return ONLY a JSON object with this exact structure:
{
    "nodes": [
        {"id": "unique_snake_case_id", "label": "Display Name", "type": "reaction|catalyst|reagent|mechanism|product|ligand|property", "description": "Brief 1-2 sentence description"}
    ],
    "edges": [
        {"source": "node_id", "target": "node_id", "label": "relationship"}
    ]
}
Rules:
- Include 8-15 nodes
- Use snake_case for IDs
- Types must be one of: reaction, catalyst, reagent, mechanism, product, ligand, property, concept, method
- Make connections educationally meaningful
- Focus on chemical accuracy
- Return ONLY the JSON, no other text"""

    user_message = f"Generate a knowledge graph for: {topic}"
    
    response = get_llm_response_with_key(system_prompt, user_message, api_key, provider)
    
    if response:
        try:
            json_str = response.strip()
            if json_str.startswith('```'):
                lines = json_str.split('\n')
                json_str = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Response was: {response[:500]}...")
    
    return None


# ==================== API ENDPOINTS ====================

@app.route('/api/knowledge-graph', methods=['POST'])
def api_knowledge_graph():
    """Generate knowledge graph for a given topic using LLM with optional API key"""
    try:
        data = request.get_json()
        topic = data.get('topic', 'cross-coupling')
        use_llm = data.get('use_llm', True)
        
        # Support dynamic API key from frontend
        api_key = data.get('api_key')
        provider = data.get('provider', 'deepseek')

        print(f"[KG API] Topic: {topic}, Use LLM: {use_llm}, Provider: {provider}")

        graph_data = None
        llm_used = False

        if use_llm:
            # Try to use LLM to generate knowledge graph
            print(f"[KG API] Attempting LLM generation for: {topic}")
            graph_data = generate_knowledge_graph_with_llm(topic, api_key, provider)
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


@app.route('/api/knowledge-graph/upload', methods=['POST'])
@limiter.limit("5 per minute")
def api_knowledge_graph_upload():
    """Upload PDF and generate knowledge graph from its content with optional API key"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded', 'success': False}), 400
        
        file = request.files['file']
        
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
        
        # Get optional API key from form data
        api_key = request.form.get('api_key')
        provider = request.form.get('provider', 'deepseek')
        
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
            graph_data = generate_kg_from_content(text_content, filename, api_key, provider)
            
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


@app.route('/api/knowledge-graph/explain', methods=['POST'])
def api_knowledge_graph_explain():
    """Get LLM explanation for a node or relationship with optional API key"""
    try:
        data = request.get_json()
        node_label = data.get('node', '')
        context = data.get('context', '')
        
        # Get optional API key
        api_key = data.get('api_key')
        provider = data.get('provider', 'deepseek')

        system_prompt = """You are an expert chemistry educator specializing in transition metal catalysis.
Provide a clear, concise explanation (2-3 sentences) for the given chemistry concept.
Focus on practical understanding and real-world applications.
Keep the explanation accessible to graduate-level chemistry students."""

        user_message = f"Explain {node_label} in the context of transition metal catalysis. Context: {context}"

        llm_response = get_llm_response_with_key(system_prompt, user_message, api_key, provider)

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


def generate_kg_from_content(content: str, source_name: str = "PDF", api_key: str = None, provider: str = None) -> dict:
    """Generate knowledge graph from text content using LLM with optional API key"""
    
    # Truncate content if too long (LLM token limits)
    max_chars = 6000
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

    response = get_llm_response_with_key(system_prompt, user_message, api_key, provider)
    
    if response:
        try:
            json_str = response.strip()
            
            # Remove markdown code blocks if present
            if '```' in json_str:
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
            
            result = json.loads(json_str)
            
            # Validate structure
            if 'nodes' in result and 'edges' in result:
                return result
            else:
                print(f"Invalid structure: missing nodes or edges")
                return None
                
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            return None
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    return None
