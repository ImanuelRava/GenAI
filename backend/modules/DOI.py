# DOI.py
import requests
import pdfplumber
import re
from typing import Tuple, List, Dict, Any, Optional

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
USER_AGENT = "CitationNetworkExplorer/1.0 (mailto:user@example.com)"
HEADERS = {'User-Agent': USER_AGENT}

# ---------------------------------------------------------
# DOI Extraction
# ---------------------------------------------------------
def extract_doi_from_pdf(pdf_path) -> str:
    """
    Extracts the first found DOI from a PDF file.
    """
    doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Z0-9]+'
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                match = re.search(doi_pattern, text, re.IGNORECASE)
                if match:
                    doi = match.group(0).strip()
                    return doi.rstrip('.;,)')
    raise ValueError("No DOI found in the PDF.")

# ---------------------------------------------------------
# Paper Details Fetching (Cached)
# ---------------------------------------------------------
def get_paper_details(doi: str) -> Tuple[str, Optional[int], int, List, str]:
    """
    Fetches author, year, citation count, references, and title for a DOI.
    Cached to minimize API calls.
    Returns: (author, year, citations, references, title)
    """
    url = f"https://api.crossref.org/works/{doi}"
    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        resp.raise_for_status()
        
        msg = resp.json()['message']

        # Year
        date_info = msg.get('published-print') or msg.get('published-online') or msg.get('published', {})
        year = date_info.get('date-parts', [[None]])[0][0]

        # Author (Last author)
        authors = msg.get('author', [])
        author = "Unknown"
        if authors:
            last_author = authors[-1]
            author = last_author.get('family') or last_author.get('name') or "Unknown"

        # Title
        title_list = msg.get('title', [])
        title = title_list[0] if title_list else "No Title"

        return author, year, msg.get('is-referenced-by-count', 0), msg.get('reference', []), title

    except requests.exceptions.HTTPError:
        raise ValueError(f"DOI not found: {doi}")
    except Exception as e:
        raise RuntimeError(f"Error fetching {doi}: {e}")

# ---------------------------------------------------------
# Cross-Reference Logic
# ---------------------------------------------------------
def get_referenced_dois(references: List) -> List[str]:
    if not references: return []
    return [ref['DOI'] for ref in references if ref and 'DOI' in ref]

def get_forward_citations(doi: str, max_papers: int = 1000) -> List[Dict[str, Any]]:
    """Fetches papers that cite the given DOI using OpenAlex API."""
    base_url = "https://api.openalex.org"
    
    # Step 1: Get OpenAlex ID for the DOI
    # We use a generic search to get the ID first to ensure the filter works
    search_url = f"{base_url}/works/doi:{doi}"
    
    try:
        resp = requests.get(search_url, timeout=15, headers=HEADERS)
        if resp.status_code != 200:
            return []
        
        work_data = resp.json()
        work_id = work_data.get('id')
        
        if not work_id:
            return []
        
        all_results = []
        page = 1
        per_page = 200
        
        while True:
            citations_url = (
                f"{base_url}/works?filter=cites:{work_id}&per_page={per_page}&page={page}"
                f"&select=id,doi,display_name,publication_year,cited_by_count,authorships,referenced_works"
            )
            
            resp = requests.get(citations_url, timeout=20, headers=HEADERS)
            if resp.status_code != 200:
                break
            
            data = resp.json()
            items = data.get('results', [])
            
            if not items:
                break
            
            for item in items:
                authors = item.get('authorships', [])
                author_name = "Unknown"
                if authors:
                    author_name = authors[0].get('author', {}).get('display_name', 'Unknown')

                all_results.append({
                    'id': item.get('id'),
                    'doi': item.get('doi'),
                    'title': item.get('display_name', 'No Title'),
                    'citations': item.get('cited_by_count', 0),
                    'year': item.get('publication_year'),
                    'author': author_name,
                    'referenced_ids': item.get('referenced_works', [])
                })

            if len(items) < per_page or len(all_results) >= max_papers:
                break
            page += 1
        return all_results
            
    except Exception as e:
        print(f"API Error: {e}")
        return []