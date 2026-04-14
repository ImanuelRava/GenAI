# Local_Reference.py
import networkx as nx
import time
from .DOI import extract_doi_from_pdf, get_paper_details, get_referenced_dois

# ---------------------------------------------------------
# Backward Citation Network
# ---------------------------------------------------------
def build_reference_network(pdf_path, progress_callback=None):
    # 1. Extract Main DOI
    try:
        main_doi = extract_doi_from_pdf(pdf_path)
        if progress_callback: progress_callback(f"Main DOI found: {main_doi}")
    except ValueError as e:
        if progress_callback: progress_callback(f"Error: {e}")
        return None, []

    G = nx.DiGraph()

    # 2. Get Main Paper Details
    try:
        main_author, main_year, main_citations, main_refs, main_title = get_paper_details(main_doi)
        G.add_node(main_doi, author=main_author or "Unknown", year=main_year or 0, 
                   citations=main_citations, is_main=True, title=main_title)
    except Exception as e:
        if progress_callback: progress_callback(f"Error fetching main paper: {e}")
        return None, []

    # 3. Get References
    ref_dois = get_referenced_dois(main_refs) if main_refs else []
    valid_refs = []
    total_refs = len(ref_dois)
    
    if progress_callback: progress_callback(f"Found {total_refs} references. Fetching details...")

    # 4. Build Network (First Pass: Direct References)
    # We use a small sleep to be polite to the API, though caching helps significantly
    for i, doi in enumerate(ref_dois):
        if progress_callback and i % 5 == 0: 
            progress_callback(f"Processing reference {i+1}/{total_refs}...")
            
        try:
            # Small polite delay if the item wasn't cached
            # (If cached, this returns instantly, making the delay negligible)
            author, year, cites, _, title = get_paper_details(doi)
            
            G.add_node(doi, author=author or "Unknown", year=year or 0, 
                       citations=cites, is_main=False, title=title)
            G.add_edge(main_doi, doi) # Main paper -> Reference
            valid_refs.append(doi)
            
            # Be nice to the API for uncached requests
            time.sleep(0.3) 
        except Exception:
            continue

    # 5. Check Cross-References (References citing each other)
    # NOTE: This can be slow. Limited to first 20 valid refs to keep UI responsive.
    # You can increase this limit if needed.
    cross_ref_limit = 20
    check_list = valid_refs[:cross_ref_limit]
    
    if progress_callback: progress_callback(f"Checking cross-references for top {len(check_list)} references...")
    
    for doi in check_list:
        try:
            _, _, _, sources, _ = get_paper_details(doi)
            cited = get_referenced_dois(sources)
            for c in cited:
                if c in valid_refs:
                    G.add_edge(doi, c) # Reference A -> Reference B
            time.sleep(0.3)
        except Exception:
            continue
    all_papers_list = []
    for n in G.nodes():
        data = G.nodes[n]
        all_papers_list.append({
            'Number': len(all_papers_list) + 1,
            'DOI': n,
            'Title': data.get('title', 'No Title'),
            'Publication Year': data.get('year', 0),
            'Corresponding Author': data.get('author', 'Unknown'),
            'Global Citation Count': data.get('citations', 0),
            'Local Citation Count': G.in_degree(n) 
        })

    # 2. Sort by Global Citations
    sorted_nodes = sorted(all_papers_list, key=lambda x: x['Global Citation Count'], reverse=True)
    
    # Get IDs of the top 30 papers
    top_30_ids = [item['DOI'] for item in sorted_nodes[:30]]

    # Ensure Main Paper is in Top 30
    if main_doi not in top_30_ids:
        top_30_ids.pop()
        top_30_ids.insert(0, main_doi)

    # 3. Create Top 30 Subgraph
    G_viz = G.subgraph(top_30_ids).copy()

    # ---------------------------------------------------------
    # SUGGESTION LOGIC (UPDATED)
    # ---------------------------------------------------------
    
    # NEW: Filter valid refs to only those in Top 30
    valid_refs_top_30 = [r for r in valid_refs if r in top_30_ids]
    
    suggestions = []
    if progress_callback: progress_callback("Generating suggestions...")

    # --- CRITERIA 1: Top Global Impact References (Only within Top 30) ---
    all_refs_data = []
    for doi in valid_refs_top_30: # Iterate ONLY top 30
        if not G.has_node(doi): continue

        node_data = G.nodes[doi]
        
        all_refs_data.append({
            'id': doi, 
            'doi': doi, 
            'title': node_data.get('title'),
            'citations': node_data.get('citations'),
            'year': node_data.get('year'),
            'author': node_data.get('author')
        })
    
    # Sort by Global Citations
    all_refs_data.sort(key=lambda x: x.get('citations', 0), reverse=True)
    
    selected_ids = set()
    for paper in all_refs_data[:5]:
        node_id = paper.get('id')
        if node_id and G.has_node(node_id):
            selected_ids.add(node_id)
            suggestions.append({
                'doi': paper.get('doi'), 
                'title': paper.get('title', 'No Title'),
                'citations': paper.get('citations', 0),
                'year': paper.get('year'),
                'author': paper.get('author'),
                'source': 'Top Global Impact Reference'
            })

    # --- CRITERIA 2: High Local Citation References (Only within Top 30) ---
    local_cite_list = []
    for doi in valid_refs_top_30: # Iterate ONLY top 30
        if doi in selected_ids:
            continue
            
        if G.has_node(doi):
            local_count = G.in_degree(doi) # Check against Full Graph G
            node_data = G.nodes[doi]
            
            paper_data = {
                'id': doi,
                'doi': doi,
                'title': node_data.get('title'),
                'citations': node_data.get('citations'),
                'year': node_data.get('year'),
                'author': node_data.get('author')
            }
            local_cite_list.append((paper_data, local_count))
    
    # Sort by Local Citations
    local_cite_list.sort(key=lambda x: x[1], reverse=True)
    
    for paper, count in local_cite_list[:5]:
        suggestions.append({
            'doi': paper.get('doi'), 
            'title': paper.get('title', 'No Title'),
            'citations': paper.get('citations', 0),
            'year': paper.get('year'),
            'author': paper.get('author'),
            'source': 'High Local Citation Reference'
        })

    return G_viz, suggestions, all_papers_list