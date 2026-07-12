import pandas as pd
import numpy as np
import networkx as nx
import logging
from .DOI import get_paper_details

logger = logging.getLogger(__name__)


def read_dois_from_excel(excel_file_like):
    try:
        df = pd.read_excel(excel_file_like)
        return df.iloc[:, 0].dropna().tolist()
    except (OSError, ValueError, KeyError) as e:
        # OSError: file not found / permission denied. ValueError: not an
        # Excel file / corrupt file. KeyError: empty DataFrame (no columns).
        logger.error(f"Error reading Excel file: {e}")
        return []


def fetch_all_details(dois, progress_callback=None):
    details, labels = {}, {}
    total = len(dois)

    for i, doi in enumerate(dois):
        if progress_callback: progress_callback(f"Fetching DOI {i+1}/{total}...")
        try:
            author, year, global_citations, references, title = get_paper_details(doi)
            ref_count = sum(1 for ref in references if ref and 'DOI' in ref)

            details[doi] = references
            labels[doi] = {
                "author": author,
                "year": year,
                "global_citations": global_citations,
                "ref_count": ref_count,
                "title": title
            }
        except (ValueError, RuntimeError) as e:
            # get_paper_details raises ValueError for DOI-not-found and
            # RuntimeError for network/parse errors. Continue to next DOI.
            logger.error(f"Error fetching details for DOI {doi}: {e}")
            continue
    return details, labels


def create_adjacency_matrix(dois, details):
    n = len(dois)
    matrix = np.zeros((n, n), dtype=int)
    doi_index = {doi: idx for idx, doi in enumerate(dois)}

    for i, doi in enumerate(dois):
        if doi in details:
            ref_dois = {ref['DOI'] for ref in details[doi] if ref and 'DOI' in ref}
            for j, target_doi in enumerate(dois):
                if i != j and target_doi in ref_dois:
                    matrix[i][j] = 1
    return matrix


def build_cross_reference_network(excel_file_like, progress_callback=None):
    dois = read_dois_from_excel(excel_file_like)
    if not dois:
        if progress_callback: progress_callback("No DOIs found in Excel.")
        return None

    details, labels = fetch_all_details(dois, progress_callback)
    valid_dois = [d for d in dois if d in labels]
    if not valid_dois: return None

    adjacency_matrix = create_adjacency_matrix(valid_dois, details)

    G = nx.DiGraph()

    for doi in valid_dois:
        data = labels[doi]
        G.add_node(doi,
                   author=data['author'],
                   year=data['year'],
                   citations=data['global_citations'],
                   ref_count=data['ref_count'],
                   title=data['title'])

    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(valid_dois[i], valid_dois[j])

    for doi in valid_dois:
        local_cit_count = G.in_degree(doi)
        G.nodes[doi]['local_citations'] = local_cit_count

    return G
