# Forward_Reference.py
import networkx as nx
import uuid
from datetime import datetime
import pandas as pd
from modules.DOI import extract_doi_from_pdf, get_paper_details, get_forward_citations

# ---------------------------------------------------------
# Forward Citation Network
# ---------------------------------------------------------
def build_forward_network(pdf_path, progress_callback=None):
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
        main_author, main_year, main_citations, _, main_title = get_paper_details(main_doi)
        G.add_node(main_doi, author=main_author or "Unknown", year=main_year or 0, 
                   citations=main_citations, is_main=True, title=main_title, type='main')
    except Exception as e:
        if progress_callback: progress_callback(f"Error fetching main paper: {e}")
        return None, []

    # 3. Get Forward Citations
    if progress_callback: progress_callback("Querying database for forward citations...")
    citing_papers = get_forward_citations(main_doi)
    
    if progress_callback: progress_callback(f"Found {len(citing_papers)} citing papers. Building network...")

    # 4. Build Network
    citing_nodes = [] 
    id_map = {} # Map OpenAlex ID -> Node ID (DOI or generated)

    # Add Citing Nodes
    for paper in citing_papers:
        doi = paper.get('doi')
        oaid = paper.get('id')
        
        # Use UUID for ID generation to guarantee uniqueness
        node_id = doi if doi else f"ref_{uuid.uuid4().hex[:8]}"
        if oaid: id_map[oaid] = node_id
            
        G.add_node(node_id, 
                   author=paper.get('author', 'Unknown'), 
                   year=paper.get('year') or 0, 
                   citations=paper.get('citations', 0), 
                   is_main=False, 
                   title=paper.get('title', 'No Title'),
                   type='citing')
        
        G.add_edge(node_id, main_doi) # Citing paper -> Main paper
        citing_nodes.append(node_id)

    # 5. Check Cross-References (Citing papers citing each other)
    if progress_callback: progress_callback("Checking cross-references (this might take a moment)...")
    
    cross_ref_count = 0
    # Optimization: Only check cross-refs for a subset to save time if list is huge
    check_papers = citing_papers[:100] if len(citing_papers) > 100 else citing_papers

    for paper in check_papers:
        source_oaid = paper.get('id')
        source_node = id_map.get(source_oaid)
        
        if not source_node: continue
            
        for ref_oaid in paper.get('referenced_ids', []):
            if ref_oaid in id_map:
                target_node = id_map[ref_oaid]
                if source_node != target_node and not G.has_edge(source_node, target_node):
                    G.add_edge(source_node, target_node)
                    cross_ref_count += 1

    if progress_callback: progress_callback(f"Found {cross_ref_count} cross-references.")
   # 1. Get all node data for the Excel file
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

    # 2. Sort by Global Citations (Descending)
    sorted_nodes = sorted(all_papers_list, key=lambda x: x['Global Citation Count'], reverse=True)
    
    # Get IDs of the top 30 papers
    top_30_ids = [item['DOI'] for item in sorted_nodes[:30]]
    
    # Ensure the Main Paper is included in Top 30 even if it has low citations
    if main_doi not in top_30_ids:
        top_30_ids.pop() # Remove the 30th paper
        top_30_ids.insert(0, main_doi) # Add main paper to top

    # 3. Create a Subgraph for Visualization containing ONLY top 30 nodes
    G_viz = G.subgraph(top_30_ids).copy()

    # ---------------------------------------------------------
    # SUGGESTION LOGIC (UPDATED)
    # ---------------------------------------------------------
    
    # NEW: Filter the candidates to ONLY the Top 30 papers
    # We use the 'id' from OpenAlex or 'doi'
    top_30_set = set(top_30_ids)
    citing_papers_filtered = [p for p in citing_papers if p.get('doi') in top_30_set or p.get('id') in top_30_set]
    
    suggestions = []
    current_year = datetime.now().year

    # CRITERIA 1: Recent High Impact (Only within Top 30)
    recent_papers = []
    for paper in citing_papers_filtered:
        year = paper.get('year')
        if year and (current_year - 5) <= year <= current_year:
            recent_papers.append(paper)
    
    # Sort by Global Citations
    recent_papers.sort(key=lambda x: x.get('citations', 0), reverse=True)
    
    selected_recent_ids = set()
    for paper in recent_papers[:5]:
        node_id = id_map.get(paper.get('id'))
        if node_id and G.has_node(node_id): # Check against Full Graph for accurate Local Citations
            selected_recent_ids.add(paper.get('id'))
            suggestions.append({
                'doi': paper.get('doi'), 
                'title': paper.get('title', 'No Title'),
                'citations': paper.get('citations', 0),
                'year': paper.get('year'),
                'author': paper.get('author'),
                'source': 'Recent High Impact Citing Paper'
            })

    # CRITERIA 2: High Local Citation (Only within Top 30)
    local_cite_list = []
    
    # We iterate over the filtered list, but check Local Citations against the Full Graph G
    for paper in citing_papers_filtered:
        pid = paper.get('id')
        if pid in selected_recent_ids:
            continue
            
        node_id = id_map.get(pid)
        if node_id and G.has_node(node_id):
            local_count = G.in_degree(node_id) # Calculate on FULL Graph
            local_cite_list.append((paper, local_count))
    
    # Sort by Local Citations
    local_cite_list.sort(key=lambda x: x[1], reverse=True)
    
    for paper, count in local_cite_list[:5]:
        node_id = id_map.get(paper.get('id'))
        suggestions.append({
            'doi': paper.get('doi'), 
            'title': paper.get('title', 'No Title'),
            'citations': paper.get('citations', 0),
            'year': paper.get('year'),
            'author': paper.get('author'),
            'source': 'High Local Citation Paper'
        })

    return G_viz, suggestions, all_papers_list