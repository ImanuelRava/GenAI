# Cross_Reference.py
import pandas as pd
import numpy as np
import networkx as nx
from modules.DOI import get_paper_details

# ---------------------------------------------------------
# Data Processing
# ---------------------------------------------------------
def read_dois_from_excel(excel_file_like):
    try:
        df = pd.read_excel(excel_file_like)
        # Assumes DOIs are in the first column
        return df.iloc[:, 0].dropna().tolist()
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

def fetch_all_details(dois, progress_callback=None):
    details, labels = {}, {}
    total = len(dois)
    
    for i, doi in enumerate(dois):
        if progress_callback: progress_callback(f"Fetching DOI {i+1}/{total}...")
        try:
            # Returns: (author, year, global_citations, references, title)
            author, year, global_citations, references, title = get_paper_details(doi)
            
            # Count how many references this paper has (Out-degree)
            ref_count = sum(1 for ref in references if ref and 'DOI' in ref) 
            
            details[doi] = references
            labels[doi] = {
                "author": author, 
                "year": year,
                "global_citations": global_citations,
                "ref_count": ref_count, 
                "title": title
            }
        except Exception as e:
            print(f"Error fetching details for DOI {doi}: {e}")
            continue
    return details, labels

# ---------------------------------------------------------
# Cross-Reference Network
# ---------------------------------------------------------
def create_adjacency_matrix(dois, details):
    n = len(dois)
    matrix = np.zeros((n, n), dtype=int)
    doi_index = {doi: idx for idx, doi in enumerate(dois)}
    
    for i, doi in enumerate(dois):
        if doi in details:
            # Get all DOIs cited by this paper
            ref_dois = {ref['DOI'] for ref in details[doi] if ref and 'DOI' in ref}
            for j, target_doi in enumerate(dois):
                # If target_doi is in our list and is cited by current doi
                if i != j and target_doi in ref_dois:
                    # matrix[i][j] = 1 means Node i cites Node j
                    matrix[i][j] = 1
    return matrix

def build_cross_reference_network(excel_file_like, progress_callback=None):
    dois = read_dois_from_excel(excel_file_like)
    if not dois:
        if progress_callback: progress_callback("No DOIs found in Excel.")
        return None

    details, labels = fetch_all_details(dois, progress_callback)
    
    # Filter out DOIs where we failed to fetch details
    valid_dois = [d for d in dois if d in labels]
    if not valid_dois: return None
        
    adjacency_matrix = create_adjacency_matrix(valid_dois, details)
    
    G = nx.DiGraph()
    
    # 1. Add Nodes
    # IMPORTANT: We store Global Citations under the key 'citations' 
    # so that utils.py (which looks for 'citations') can plot it correctly.
    for doi in valid_dois:
        data = labels[doi]
        G.add_node(doi, 
                   author=data['author'], 
                   year=data['year'], 
                   citations=data['global_citations'], # Fixed key for utils.py
                   ref_count=data['ref_count'], 
                   title=data['title'])

    # 2. Add Edges
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i][j] == 1:
                # Node i references Node j
                G.add_edge(valid_dois[i], valid_dois[j])
    
    # 3. Calculate Local Citations
    # Local Citations = How many times this paper is cited by OTHER papers in the Excel file
    # This is equivalent to the In-Degree of the node in the graph.
    for doi in valid_dois:
        local_cit_count = G.in_degree(doi)
        G.nodes[doi]['local_citations'] = local_cit_count
        
    return G