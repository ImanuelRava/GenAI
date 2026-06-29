"""
Tests for the citation-network pipeline.

Covers:
  - modules/DOI.py            — extract_doi_from_pdf, get_paper_details,
                                get_referenced_dois, get_forward_citations
  - modules/Cross_Reference.py — read_dois_from_excel, fetch_all_details,
                                create_adjacency_matrix, build_cross_reference_network
  - modules/Backward_Reference.py — build_reference_network
  - modules/Forward_Reference.py  — build_forward_network
  - routes/network.py         — validate_file_upload, _get_extension,
                                POST /api/network endpoint

All external HTTP calls (Crossref, OpenAlex) are mocked. PDF and Excel
fixtures are created as real temp files so pdfplumber and pandas can
read them.
"""

import sys
import os
import io
import json
import tempfile
from unittest.mock import patch, MagicMock, mock_open

import pytest

# Make backend/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


# =============================================================================
# Shared fixtures + helpers
# =============================================================================

@pytest.fixture
def sample_pdf_with_doi(tmp_path):
    """Create a minimal real PDF file containing a DOI string.

    We can't easily create a valid PDF from scratch in a test, so we mock
    pdfplumber.open to return a fake page with the DOI text. The pdf_path
    passed to extract_doi_from_pdf doesn't need to exist on disk because
    we mock pdfplumber.open.
    """
    return str(tmp_path / "fake_paper.pdf")


@pytest.fixture
def sample_excel_with_dois(tmp_path):
    """Create a real .xlsx file containing a column of DOIs.

    Uses pandas + openpyxl to write a genuine Excel file that
    pd.read_excel can read back.
    """
    import pandas as pd
    excel_path = tmp_path / "dois.xlsx"
    df = pd.DataFrame({
        'DOI': [
            '10.1000/aaa',
            '10.1000/bbb',
            '10.1000/ccc',
        ],
    })
    df.to_excel(excel_path, index=False)
    return str(excel_path)


def _make_crossref_response(doi, title="Test Paper", year=2020, author="Smith",
                             citations=10, references=None):
    """Build a mock Crossref API response JSON for get_paper_details."""
    return {
        'message': {
            'title': [title],
            'published-print': {'date-parts': [[year]]},
            'author': [{'family': author, 'given': 'J.'}],
            'is-referenced-by-count': citations,
            'reference': references or [],
        }
    }


def _make_openalex_work_response(work_id="W123", doi="10.1000/aaa"):
    """Build a mock OpenAlex /works/doi:XXX response for get_forward_citations."""
    return {
        'id': work_id,
        'doi': doi,
    }


def _make_openalex_citations_response(citing_papers):
    """Build a mock OpenAlex /works?filter=cites:XXX response.

    citing_papers: list of dicts with keys (id, doi, display_name,
    publication_year, cited_by_count, authorships, referenced_works).
    """
    return {
        'results': citing_papers,
        'meta': {'count': len(citing_papers)},
    }


# =============================================================================
# modules/DOI.py — extract_doi_from_pdf
# =============================================================================

class TestExtractDoiFromPdf:
    """Tests for extract_doi_from_pdf()."""

    def _mock_pdfplumber(self, pages_text):
        """Build a mock pdfplumber.open context manager.

        pages_text: list of strings (or None) — one per page.
        """
        mock_pages = []
        for text in pages_text:
            page = MagicMock()
            page.extract_text.return_value = text
            mock_pages.append(page)

        mock_pdf = MagicMock()
        mock_pdf.pages = mock_pages

        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_pdf)
        mock_context.__exit__ = MagicMock(return_value=None)
        return mock_context

    def test_extracts_doi_from_pdf_text(self, sample_pdf_with_doi):
        """A PDF containing 'doi: 10.1038/nature12373' should return that DOI."""
        from modules.DOI import extract_doi_from_pdf

        with patch('modules.DOI.pdfplumber.open',
                   return_value=self._mock_pdfplumber([
                       "This is a paper. DOI: 10.1038/nature12373. Published in Nature."
                   ])):
            doi = extract_doi_from_pdf(sample_pdf_with_doi)
            assert doi == '10.1038/nature12373'

    def test_strips_trailing_punctuation(self, sample_pdf_with_doi):
        """DOIs matched with trailing '.' or ')' should be stripped."""
        from modules.DOI import extract_doi_from_pdf

        with patch('modules.DOI.pdfplumber.open',
                   return_value=self._mock_pdfplumber([
                       "See ref 10.1000/test.) for details"
                   ])):
            doi = extract_doi_from_pdf(sample_pdf_with_doi)
            assert doi == '10.1000/test'

    def test_searches_multiple_pages(self, sample_pdf_with_doi):
        """If the first page has no DOI, the second page should be searched."""
        from modules.DOI import extract_doi_from_pdf

        with patch('modules.DOI.pdfplumber.open',
                   return_value=self._mock_pdfplumber([
                       "Title page with no DOI here.",
                       "References: 10.1234/abcd1234",
                   ])):
            doi = extract_doi_from_pdf(sample_pdf_with_doi)
            assert doi == '10.1234/abcd1234'

    def test_raises_value_error_when_no_doi_found(self, sample_pdf_with_doi):
        """If no DOI is found on any page, ValueError should be raised."""
        from modules.DOI import extract_doi_from_pdf

        with patch('modules.DOI.pdfplumber.open',
                   return_value=self._mock_pdfplumber(["No DOI in this text."])):
            with pytest.raises(ValueError, match='No DOI found'):
                extract_doi_from_pdf(sample_pdf_with_doi)

    def test_skips_pages_with_no_text(self, sample_pdf_with_doi):
        """Pages where extract_text returns None or empty should be skipped."""
        from modules.DOI import extract_doi_from_pdf

        with patch('modules.DOI.pdfplumber.open',
                   return_value=self._mock_pdfplumber([None, "", "DOI: 10.9999/found"])):
            doi = extract_doi_from_pdf(sample_pdf_with_doi)
            assert doi == '10.9999/found'

    def test_matches_doi_case_insensitively(self, sample_pdf_with_doi):
        """The DOI regex uses re.IGNORECASE."""
        from modules.DOI import extract_doi_from_pdf

        with patch('modules.DOI.pdfplumber.open',
                   return_value=self._mock_pdfplumber(["DOI: 10.1038/ABCDEF"])):
            doi = extract_doi_from_pdf(sample_pdf_with_doi)
            assert '10.1038/ABCDEF' in doi


# =============================================================================
# modules/DOI.py — get_paper_details
# =============================================================================

class TestGetPaperDetails:
    """Tests for get_paper_details()."""

    def test_returns_tuple_of_five_elements(self):
        """get_paper_details returns (author, year, citations, references, title)."""
        from modules.DOI import get_paper_details

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_crossref_response(
            '10.1000/test', title='My Paper', year=2021, author='Doe',
            citations=42, references=[{'DOI': '10.2000/ref1'}],
        )
        mock_response.raise_for_status = MagicMock()

        with patch('modules.DOI.requests.get', return_value=mock_response):
            result = get_paper_details('10.1000/test')
            assert len(result) == 5
            author, year, citations, references, title = result
            assert author == 'Doe'
            assert year == 2021
            assert citations == 42
            assert references == [{'DOI': '10.2000/ref1'}]
            assert title == 'My Paper'

    def test_falls_back_to_published_online_date(self):
        """If published-print is missing, published-online should be used."""
        from modules.DOI import get_paper_details

        response_data = {
            'message': {
                'title': ['Online Paper'],
                'published-online': {'date-parts': [[2019]]},
                'author': [{'family': 'OnlineAuthor'}],
                'is-referenced-by-count': 5,
                'reference': [],
            }
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        with patch('modules.DOI.requests.get', return_value=mock_response):
            _, year, _, _, title = get_paper_details('10.1000/online')
            assert year == 2019
            assert title == 'Online Paper'

    def test_handles_missing_author(self):
        """If no author field, return 'Unknown'."""
        from modules.DOI import get_paper_details

        response_data = {
            'message': {
                'title': ['No Author Paper'],
                'published-print': {'date-parts': [[2020]]},
                'is-referenced-by-count': 0,
                'reference': [],
            }
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        with patch('modules.DOI.requests.get', return_value=mock_response):
            author, _, _, _, _ = get_paper_details('10.1000/noauthor')
            assert author == 'Unknown'

    def test_handles_empty_title_list(self):
        """If title list is empty, return 'No Title'."""
        from modules.DOI import get_paper_details

        response_data = {
            'message': {
                'title': [],
                'published-print': {'date-parts': [[2020]]},
                'author': [{'family': 'X'}],
                'is-referenced-by-count': 0,
                'reference': [],
            }
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        with patch('modules.DOI.requests.get', return_value=mock_response):
            _, _, _, _, title = get_paper_details('10.1000/notitle')
            assert title == 'No Title'

    def test_raises_value_error_on_404(self):
        """HTTP 404 → ValueError (DOI not found)."""
        import requests
        from modules.DOI import get_paper_details

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = (
            requests.exceptions.HTTPError(response=MagicMock(status_code=404))
        )

        with patch('modules.DOI.requests.get', return_value=mock_response):
            with pytest.raises(ValueError, match='DOI not found'):
                get_paper_details('10.1000/nonexistent')

    def test_raises_runtime_error_on_timeout(self):
        """Timeout → RuntimeError."""
        import requests
        from modules.DOI import get_paper_details

        with patch('modules.DOI.requests.get',
                   side_effect=requests.exceptions.Timeout()):
            with pytest.raises(RuntimeError, match='Request timeout'):
                get_paper_details('10.1000/timeout')

    def test_raises_runtime_error_on_connection_error(self):
        """ConnectionError → RuntimeError (wrapped in the catch-all)."""
        import requests
        from modules.DOI import get_paper_details

        with patch('modules.DOI.requests.get',
                   side_effect=requests.exceptions.ConnectionError()):
            with pytest.raises(RuntimeError, match='Error fetching'):
                get_paper_details('10.1000/connerror')

    def test_raises_runtime_error_on_json_decode_error(self):
        """Invalid JSON response → RuntimeError."""
        from modules.DOI import get_paper_details

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("invalid JSON")
        mock_response.raise_for_status = MagicMock()

        with patch('modules.DOI.requests.get', return_value=mock_response):
            with pytest.raises(RuntimeError, match='Error fetching'):
                get_paper_details('10.1000/badjson')

    def test_uses_last_author_not_first(self):
        """The code uses authors[-1] (last author), not authors[0]."""
        from modules.DOI import get_paper_details

        response_data = {
            'message': {
                'title': ['Multi-Author Paper'],
                'published-print': {'date-parts': [[2020]]},
                'author': [
                    {'family': 'FirstAuthor'},
                    {'family': 'MiddleAuthor'},
                    {'family': 'LastAuthor'},
                ],
                'is-referenced-by-count': 10,
                'reference': [],
            }
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        with patch('modules.DOI.requests.get', return_value=mock_response):
            author, _, _, _, _ = get_paper_details('10.1000/multi')
            assert author == 'LastAuthor'


# =============================================================================
# modules/DOI.py — get_referenced_dois
# =============================================================================

class TestGetReferencedDois:
    """Tests for get_referenced_dois()."""

    def test_empty_references_returns_empty_list(self):
        from modules.DOI import get_referenced_dois
        assert get_referenced_dois([]) == []

    def test_none_references_returns_empty_list(self):
        from modules.DOI import get_referenced_dois
        assert get_referenced_dois(None) == []

    def test_extracts_dois_from_references(self):
        from modules.DOI import get_referenced_dois
        refs = [
            {'DOI': '10.1000/a'},
            {'DOI': '10.1000/b'},
            {'DOI': '10.1000/c'},
        ]
        result = get_referenced_dois(refs)
        assert result == ['10.1000/a', '10.1000/b', '10.1000/c']

    def test_skips_references_without_doi_key(self):
        """References without a 'DOI' key should be skipped."""
        from modules.DOI import get_referenced_dois
        refs = [
            {'DOI': '10.1000/a'},
            {'unstructured-text': 'Some citation'},  # no DOI key
            {'DOI': '10.1000/b'},
            None,  # None entry should be skipped
        ]
        result = get_referenced_dois(refs)
        assert result == ['10.1000/a', '10.1000/b']


# =============================================================================
# modules/DOI.py — get_forward_citations
# =============================================================================

class TestGetForwardCitations:
    """Tests for get_forward_citations()."""

    def test_returns_empty_list_on_non_200_status(self):
        """If OpenAlex returns non-200 for the initial work lookup, return []."""
        from modules.DOI import get_forward_citations

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch('modules.DOI.requests.get', return_value=mock_response):
            result = get_forward_citations('10.1000/nonexistent')
            assert result == []

    def test_returns_empty_list_when_no_work_id(self):
        """If the OpenAlex response has no 'id' field, return []."""
        from modules.DOI import get_forward_citations

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'doi': '10.1000/aaa'}  # no 'id'

        with patch('modules.DOI.requests.get', return_value=mock_response):
            result = get_forward_citations('10.1000/aaa')
            assert result == []

    def test_returns_citing_papers_list(self):
        """A successful response should return a list of citing paper dicts."""
        from modules.DOI import get_forward_citations

        work_response = MagicMock()
        work_response.status_code = 200
        work_response.json.return_value = _make_openalex_work_response(
            work_id='W123', doi='10.1000/main',
        )

        citing_paper = {
            'id': 'W456',
            'doi': '10.2000/citing1',
            'display_name': 'Citing Paper 1',
            'publication_year': 2021,
            'cited_by_count': 5,
            'authorships': [{'author': {'display_name': 'Citer'}}],
            'referenced_works': ['W123'],
        }
        citations_response = MagicMock()
        citations_response.status_code = 200
        citations_response.json.return_value = _make_openalex_citations_response([citing_paper])

        with patch('modules.DOI.requests.get',
                   side_effect=[work_response, citations_response]):
            result = get_forward_citations('10.1000/main')
            assert len(result) == 1
            paper = result[0]
            assert paper['doi'] == '10.2000/citing1'
            assert paper['title'] == 'Citing Paper 1'
            assert paper['citations'] == 5
            assert paper['year'] == 2021
            assert paper['author'] == 'Citer'
            assert paper['referenced_ids'] == ['W123']

    def test_handles_citing_paper_with_no_authorships(self):
        """If a citing paper has no authorships, author should be 'Unknown'."""
        from modules.DOI import get_forward_citations

        work_response = MagicMock()
        work_response.status_code = 200
        work_response.json.return_value = _make_openalex_work_response()

        citing_paper = {
            'id': 'W789',
            'doi': '10.2000/noauthors',
            'display_name': 'Anon Paper',
            'publication_year': 2022,
            'cited_by_count': 0,
            'authorships': [],  # empty
            'referenced_works': [],
        }
        citations_response = MagicMock()
        citations_response.status_code = 200
        citations_response.json.return_value = _make_openalex_citations_response([citing_paper])

        with patch('modules.DOI.requests.get',
                   side_effect=[work_response, citations_response]):
            result = get_forward_citations('10.1000/main')
            assert result[0]['author'] == 'Unknown'

    def test_returns_empty_list_on_network_error(self):
        """Network errors should return [] (not raise)."""
        import requests
        from modules.DOI import get_forward_citations

        with patch('modules.DOI.requests.get',
                   side_effect=requests.exceptions.ConnectionError()):
            result = get_forward_citations('10.1000/fail')
            assert result == []

    def test_stops_when_results_page_is_empty(self):
        """Pagination should stop when an empty results page is returned."""
        from modules.DOI import get_forward_citations

        work_response = MagicMock()
        work_response.status_code = 200
        work_response.json.return_value = _make_openalex_work_response()

        empty_citations = MagicMock()
        empty_citations.status_code = 200
        empty_citations.json.return_value = {'results': []}

        with patch('modules.DOI.requests.get',
                   side_effect=[work_response, empty_citations]):
            result = get_forward_citations('10.1000/main')
            assert result == []


# =============================================================================
# modules/Cross_Reference.py — read_dois_from_excel
# =============================================================================

class TestReadDoisFromExcel:
    """Tests for read_dois_from_excel()."""

    def test_reads_dois_from_real_excel_file(self, sample_excel_with_dois):
        """A real .xlsx file with a DOI column should return the DOI list."""
        from modules.Cross_Reference import read_dois_from_excel
        with open(sample_excel_with_dois, 'rb') as f:
            # read_dois_from_excel accepts a file-like object
            from io import BytesIO
            f.seek(0)
            result = read_dois_from_excel(BytesIO(f.read()))
        assert result == ['10.1000/aaa', '10.1000/bbb', '10.1000/ccc']

    def test_reads_dois_from_file_path(self, sample_excel_with_dois):
        """read_dois_from_excel also accepts a file path string."""
        from modules.Cross_Reference import read_dois_from_excel
        result = read_dois_from_excel(sample_excel_with_dois)
        assert result == ['10.1000/aaa', '10.1000/bbb', '10.1000/ccc']

    def test_returns_empty_list_on_corrupt_file(self, tmp_path):
        """A non-Excel file should return [] (ValueError caught)."""
        from modules.Cross_Reference import read_dois_from_excel
        bad_file = tmp_path / "not_excel.txt"
        bad_file.write_text("this is not an Excel file")
        result = read_dois_from_excel(str(bad_file))
        assert result == []

    def test_returns_empty_list_on_nonexistent_file(self):
        """A non-existent file path should return [] (OSError caught)."""
        from modules.Cross_Reference import read_dois_from_excel
        result = read_dois_from_excel('/nonexistent/path/file.xlsx')
        assert result == []

    def test_drops_na_values(self, tmp_path):
        """NaN values in the first column should be dropped."""
        import pandas as pd
        from modules.Cross_Reference import read_dois_from_excel
        excel_path = tmp_path / "with_na.xlsx"
        df = pd.DataFrame({'DOI': ['10.1/a', None, '10.1/b', float('nan')]})
        df.to_excel(excel_path, index=False)
        result = read_dois_from_excel(str(excel_path))
        assert result == ['10.1/a', '10.1/b']


# =============================================================================
# modules/Cross_Reference.py — fetch_all_details
# =============================================================================

class TestFetchAllDetails:
    """Tests for fetch_all_details()."""

    def test_fetches_details_for_all_dois(self):
        """Each DOI should be fetched and stored in the details + labels dicts."""
        from modules.Cross_Reference import fetch_all_details

        def mock_get_paper_details(doi):
            return ('Author', 2020, 5, [{'DOI': '10.1/ref'}], 'Title')

        with patch('modules.Cross_Reference.get_paper_details',
                   side_effect=mock_get_paper_details):
            details, labels = fetch_all_details(['10.1/a', '10.1/b'])

        assert set(details.keys()) == {'10.1/a', '10.1/b'}
        assert set(labels.keys()) == {'10.1/a', '10.1/b'}
        assert labels['10.1/a']['author'] == 'Author'
        assert labels['10.1/a']['ref_count'] == 1  # one ref with DOI

    def test_skips_dois_that_raise_errors(self):
        """DOIs that raise ValueError or RuntimeError should be skipped."""
        from modules.Cross_Reference import fetch_all_details

        def mock_get_paper_details(doi):
            if doi == '10.1/bad':
                raise ValueError('DOI not found')
            return ('Author', 2020, 5, [], 'Title')

        with patch('modules.Cross_Reference.get_paper_details',
                   side_effect=mock_get_paper_details):
            details, labels = fetch_all_details(['10.1/good', '10.1/bad'])

        assert '10.1/good' in labels
        assert '10.1/bad' not in labels

    def test_invokes_progress_callback(self):
        """The progress_callback should be called with status messages."""
        from modules.Cross_Reference import fetch_all_details

        progress_messages = []
        def callback(msg):
            progress_messages.append(msg)

        with patch('modules.Cross_Reference.get_paper_details',
                   return_value=('A', 2020, 1, [], 'T')):
            fetch_all_details(['10.1/a', '10.1/b'], progress_callback=callback)

        assert len(progress_messages) >= 2
        assert any('1/2' in m for m in progress_messages)
        assert any('2/2' in m for m in progress_messages)

    def test_counts_only_refs_with_doi_key(self):
        """ref_count should only count references that have a 'DOI' key."""
        from modules.Cross_Reference import fetch_all_details

        refs = [
            {'DOI': '10.1/x'},      # counted
            {'unstructured': '...'}, # not counted
            {'DOI': '10.1/y'},      # counted
            None,                    # not counted
        ]
        with patch('modules.Cross_Reference.get_paper_details',
                   return_value=('A', 2020, 1, refs, 'T')):
            _, labels = fetch_all_details(['10.1/a'])

        assert labels['10.1/a']['ref_count'] == 2


# =============================================================================
# modules/Cross_Reference.py — create_adjacency_matrix
# =============================================================================

class TestCreateAdjacencyMatrix:
    """Tests for create_adjacency_matrix()."""

    def test_empty_dois_returns_empty_matrix(self):
        """No DOIs → 0x0 matrix."""
        import numpy as np
        from modules.Cross_Reference import create_adjacency_matrix
        matrix = create_adjacency_matrix([], {})
        assert matrix.shape == (0, 0)

    def test_no_cross_references_returns_zero_matrix(self):
        """If no DOI references another, the matrix is all zeros."""
        import numpy as np
        from modules.Cross_Reference import create_adjacency_matrix

        dois = ['10.1/a', '10.1/b', '10.1/c']
        details = {
            '10.1/a': [{'DOI': '10.999/other'}],  # not in dois list
            '10.1/b': [],
            '10.1/c': [],
        }
        matrix = create_adjacency_matrix(dois, details)
        assert matrix.shape == (3, 3)
        assert matrix.sum() == 0

    def test_sets_matrix_1_for_cross_references(self):
        """If DOI A references DOI B (both in list), matrix[A][B] = 1."""
        import numpy as np
        from modules.Cross_Reference import create_adjacency_matrix

        dois = ['10.1/a', '10.1/b', '10.1/c']
        details = {
            '10.1/a': [{'DOI': '10.1/b'}, {'DOI': '10.1/c'}],  # a → b, a → c
            '10.1/b': [{'DOI': '10.1/c'}],                      # b → c
            '10.1/c': [],
        }
        matrix = create_adjacency_matrix(dois, details)
        # a (row 0) references b (col 1) and c (col 2)
        assert matrix[0][1] == 1
        assert matrix[0][2] == 1
        # b (row 1) references c (col 2)
        assert matrix[1][2] == 1
        # No self-references
        assert matrix[0][0] == 0
        assert matrix[1][1] == 0
        assert matrix[2][2] == 0
        # No reverse references
        assert matrix[1][0] == 0
        assert matrix[2][0] == 0
        assert matrix[2][1] == 0

    def test_ignores_dois_not_in_details(self):
        """DOIs missing from the details dict should produce all-zero rows."""
        import numpy as np
        from modules.Cross_Reference import create_adjacency_matrix

        dois = ['10.1/a', '10.1/b']
        details = {
            '10.1/a': [{'DOI': '10.1/b'}],
            # 10.1/b missing from details
        }
        matrix = create_adjacency_matrix(dois, details)
        assert matrix[0][1] == 1  # a → b
        assert matrix[1][0] == 0  # b has no refs (missing from details)

    def test_skips_references_without_doi_key(self):
        """References without a 'DOI' key should be ignored."""
        from modules.Cross_Reference import create_adjacency_matrix

        dois = ['10.1/a', '10.1/b']
        details = {
            '10.1/a': [{'unstructured': 'text'}, {'DOI': '10.1/b'}],
        }
        matrix = create_adjacency_matrix(dois, details)
        assert matrix[0][1] == 1


# =============================================================================
# modules/Cross_Reference.py — build_cross_reference_network (integration)
# =============================================================================

class TestBuildCrossReferenceNetwork:
    """Integration tests for build_cross_reference_network()."""

    def test_returns_none_when_no_dois_in_excel(self, tmp_path):
        """An Excel file with no DOIs should return None."""
        import pandas as pd
        from modules.Cross_Reference import build_cross_reference_network

        excel_path = tmp_path / "empty.xlsx"
        pd.DataFrame({'NotDOIs': ['a', 'b']}).to_excel(excel_path, index=False)

        result = build_cross_reference_network(str(excel_path))
        # The first column is read regardless of header name, so DOIs will
        # be ['a', 'b']. But get_paper_details will fail for these non-DOIs,
        # so valid_dois will be empty → return None.
        with patch('modules.Cross_Reference.get_paper_details',
                   side_effect=ValueError('not a DOI')):
            result = build_cross_reference_network(str(excel_path))
        assert result is None

    def test_builds_network_with_cross_references(self, sample_excel_with_dois):
        """A valid Excel file with cross-referencing DOIs should produce a
        networkx.DiGraph with nodes and edges."""
        from modules.Cross_Reference import build_cross_reference_network
        import networkx as nx

        # Mock get_paper_details to return references that cross-reference
        def mock_details(doi):
            refs = []
            if doi == '10.1000/aaa':
                refs = [{'DOI': '10.1000/bbb'}]  # aaa → bbb
            elif doi == '10.1000/bbb':
                refs = [{'DOI': '10.1000/ccc'}]  # bbb → ccc
            return ('Author', 2020, 10, refs, f'Title-{doi}')

        with patch('modules.Cross_Reference.get_paper_details',
                   side_effect=mock_details):
            G = build_cross_reference_network(sample_excel_with_dois)

        assert G is not None
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 3
        # aaa → bbb, bbb → ccc
        assert G.has_edge('10.1000/aaa', '10.1000/bbb')
        assert G.has_edge('10.1000/bbb', '10.1000/ccc')
        # No reverse edges
        assert not G.has_edge('10.1000/bbb', '10.1000/aaa')
        # Node attributes set correctly
        node_data = G.nodes['10.1000/aaa']
        assert node_data['author'] == 'Author'
        assert node_data['year'] == 2020
        assert node_data['citations'] == 10
        assert 'local_citations' in node_data

    def test_sets_local_citations_correctly(self, sample_excel_with_dois):
        """local_citations = in_degree of each node."""
        from modules.Cross_Reference import build_cross_reference_network

        def mock_details(doi):
            refs = []
            if doi == '10.1000/aaa':
                refs = [{'DOI': '10.1000/bbb'}, {'DOI': '10.1000/ccc'}]
            return ('A', 2020, 5, refs, 'T')

        with patch('modules.Cross_Reference.get_paper_details',
                   side_effect=mock_details):
            G = build_cross_reference_network(sample_excel_with_dois)

        # aaa has in_degree 0 (nothing references it)
        assert G.nodes['10.1000/aaa']['local_citations'] == 0
        # bbb has in_degree 1 (aaa references it)
        assert G.nodes['10.1000/bbb']['local_citations'] == 1
        # ccc has in_degree 1 (aaa references it)
        assert G.nodes['10.1000/ccc']['local_citations'] == 1

    def test_invokes_progress_callback(self, sample_excel_with_dois):
        """The progress_callback should be called during the build."""
        from modules.Cross_Reference import build_cross_reference_network

        messages = []
        def callback(msg):
            messages.append(msg)

        with patch('modules.Cross_Reference.get_paper_details',
                   return_value=('A', 2020, 1, [], 'T')):
            build_cross_reference_network(sample_excel_with_dois, progress_callback=callback)

        assert len(messages) > 0
        assert any('Fetching' in m for m in messages)


# =============================================================================
# modules/Backward_Reference.py — build_reference_network
# =============================================================================

class TestBuildReferenceNetwork:
    """Tests for build_reference_network()."""

    def test_returns_none_when_doi_extraction_fails(self, sample_pdf_with_doi):
        """If no DOI can be extracted from the PDF, return (None, []).

        Note: the error path returns a 2-tuple (None, []), while the success
        path returns a 3-tuple (G, suggestions, all_papers). This is a
        pre-existing inconsistency in the original code — callers must handle
        both shapes.
        """
        from modules.Backward_Reference import build_reference_network

        with patch('modules.Backward_Reference.extract_doi_from_pdf',
                   side_effect=ValueError('No DOI found')):
            result = build_reference_network(sample_pdf_with_doi)
            assert len(result) == 2
            assert result[0] is None
            assert result[1] == []

    def test_returns_none_when_main_paper_fetch_fails(self, sample_pdf_with_doi):
        """If get_paper_details fails for the main DOI, return (None, [])."""
        from modules.Backward_Reference import build_reference_network

        with patch('modules.Backward_Reference.extract_doi_from_pdf',
                   return_value='10.1000/main'), \
             patch('modules.Backward_Reference.get_paper_details',
                   side_effect=RuntimeError('network error')):
            result = build_reference_network(sample_pdf_with_doi)
            assert len(result) == 2
            assert result[0] is None

    def test_builds_network_with_references(self, sample_pdf_with_doi):
        """A successful extraction should build a DiGraph with the main paper
        and its references."""
        from modules.Backward_Reference import build_reference_network
        import networkx as nx

        main_refs = [{'DOI': '10.1/ref1'}, {'DOI': '10.1/ref2'}]

        def mock_details(doi):
            if doi == '10.1000/main':
                return ('MainAuthor', 2020, 50, main_refs, 'Main Paper')
            return ('RefAuthor', 2019, 5, [], f'Ref {doi}')

        with patch('modules.Backward_Reference.extract_doi_from_pdf',
                   return_value='10.1000/main'), \
             patch('modules.Backward_Reference.get_paper_details',
                   side_effect=mock_details), \
             patch('modules.Backward_Reference.time.sleep'):  # skip sleeps
            G, suggestions, all_papers = build_reference_network(sample_pdf_with_doi)

        assert G is not None
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() >= 1  # at least the main paper
        assert '10.1000/main' in G.nodes
        # The main paper should have is_main=True
        assert G.nodes['10.1000/main']['is_main'] is True
        # References should be nodes with edges from main
        assert G.has_edge('10.1000/main', '10.1/ref1')
        assert G.has_edge('10.1000/main', '10.1/ref2')
        # all_papers list should include all nodes
        assert len(all_papers) == G.number_of_nodes()

    def test_skips_references_that_fail_fetch(self, sample_pdf_with_doi):
        """References that can't be fetched should be skipped (not crash)."""
        from modules.Backward_Reference import build_reference_network

        main_refs = [{'DOI': '10.1/good'}, {'DOI': '10.1/bad'}]

        def mock_details(doi):
            if doi == '10.1000/main':
                return ('A', 2020, 50, main_refs, 'Main')
            if doi == '10.1/bad':
                raise ValueError('not found')
            return ('A', 2019, 5, [], 'Good Ref')

        with patch('modules.Backward_Reference.extract_doi_from_pdf',
                   return_value='10.1000/main'), \
             patch('modules.Backward_Reference.get_paper_details',
                   side_effect=mock_details), \
             patch('modules.Backward_Reference.time.sleep'):
            G, suggestions, all_papers = build_reference_network(sample_pdf_with_doi)

        assert G is not None
        assert '10.1/good' in G.nodes
        assert '10.1/bad' not in G.nodes  # bad ref skipped

    def test_progress_callback_invoked(self, sample_pdf_with_doi):
        """The progress_callback should be called with status messages."""
        from modules.Backward_Reference import build_reference_network

        messages = []
        def callback(msg):
            messages.append(msg)

        with patch('modules.Backward_Reference.extract_doi_from_pdf',
                   return_value='10.1000/main'), \
             patch('modules.Backward_Reference.get_paper_details',
                   return_value=('A', 2020, 50, [], 'T')), \
             patch('modules.Backward_Reference.time.sleep'):
            build_reference_network(sample_pdf_with_doi, progress_callback=callback)

        assert any('Main DOI found' in m for m in messages)
        assert any('Found 0 references' in m for m in messages)

    def test_suggestions_have_source_field(self, sample_pdf_with_doi):
        """Each suggestion should have a 'source' field describing its category."""
        from modules.Backward_Reference import build_reference_network

        main_refs = [{'DOI': f'10.1/ref{i}'} for i in range(10)]

        def mock_details(doi):
            if doi == '10.1000/main':
                return ('A', 2020, 100, main_refs, 'Main')
            # Extract ref number from DOI for varying citation counts
            ref_num = int(doi.split('ref')[-1]) if 'ref' in doi else 0
            return ('A', 2019, ref_num * 10, [], f'Ref {doi}')

        with patch('modules.Backward_Reference.extract_doi_from_pdf',
                   return_value='10.1000/main'), \
             patch('modules.Backward_Reference.get_paper_details',
                   side_effect=mock_details), \
             patch('modules.Backward_Reference.time.sleep'):
            _, suggestions, _ = build_reference_network(sample_pdf_with_doi)

        for s in suggestions:
            assert 'source' in s
            assert s['source'] in (
                'Top Global Impact Reference',
                'High Local Citation Reference',
            )


# =============================================================================
# modules/Forward_Reference.py — build_forward_network
# =============================================================================

class TestBuildForwardNetwork:
    """Tests for build_forward_network()."""

    def test_returns_none_when_doi_extraction_fails(self, sample_pdf_with_doi):
        """If no DOI can be extracted, return (None, []).

        Note: the error path returns a 2-tuple, while the success path
        returns a 3-tuple. This is a pre-existing inconsistency.
        """
        from modules.Forward_Reference import build_forward_network

        with patch('modules.Forward_Reference.extract_doi_from_pdf',
                   side_effect=ValueError('No DOI found')):
            result = build_forward_network(sample_pdf_with_doi)
            assert len(result) == 2
            assert result[0] is None
            assert result[1] == []

    def test_returns_none_when_main_paper_fetch_fails(self, sample_pdf_with_doi):
        """If get_paper_details fails for the main DOI, return (None, [])."""
        from modules.Forward_Reference import build_forward_network

        with patch('modules.Forward_Reference.extract_doi_from_pdf',
                   return_value='10.1000/main'), \
             patch('modules.Forward_Reference.get_paper_details',
                   side_effect=RuntimeError('network error')):
            result = build_forward_network(sample_pdf_with_doi)
            assert len(result) == 2
            assert result[0] is None

    def test_builds_network_with_citing_papers(self, sample_pdf_with_doi):
        """A successful extraction should build a DiGraph with citing papers
        pointing TO the main paper."""
        from modules.Forward_Reference import build_forward_network
        import networkx as nx

        citing_papers = [
            {
                'id': 'W1', 'doi': '10.1/citing1',
                'display_name': 'Citing Paper 1',
                'publication_year': 2021, 'cited_by_count': 3,
                'authorships': [{'author': {'display_name': 'Citer1'}}],
                'referenced_works': [],
            },
            {
                'id': 'W2', 'doi': '10.1/citing2',
                'display_name': 'Citing Paper 2',
                'publication_year': 2022, 'cited_by_count': 7,
                'authorships': [{'author': {'display_name': 'Citer2'}}],
                'referenced_works': [],
            },
        ]

        with patch('modules.Forward_Reference.extract_doi_from_pdf',
                   return_value='10.1000/main'), \
             patch('modules.Forward_Reference.get_paper_details',
                   return_value=('MainAuthor', 2020, 50, [], 'Main Paper')), \
             patch('modules.Forward_Reference.get_forward_citations',
                   return_value=citing_papers):
            G, suggestions, all_papers = build_forward_network(sample_pdf_with_doi)

        assert G is not None
        assert isinstance(G, nx.DiGraph)
        assert '10.1000/main' in G.nodes
        assert G.nodes['10.1000/main']['is_main'] is True
        assert G.nodes['10.1000/main']['type'] == 'main'
        # Citing papers point TO main (edge direction: citing → main)
        assert G.has_edge('10.1/citing1', '10.1000/main')
        assert G.has_edge('10.1/citing2', '10.1000/main')

    def test_handles_citing_paper_without_doi(self, sample_pdf_with_doi):
        """Citing papers without a DOI should get a generated ref_XXX node id."""
        from modules.Forward_Reference import build_forward_network

        citing_papers = [
            {
                'id': 'W1', 'doi': None,  # no DOI
                'display_name': 'Anon Citing',
                'publication_year': 2021, 'cited_by_count': 0,
                'authorships': [], 'referenced_works': [],
            },
        ]

        with patch('modules.Forward_Reference.extract_doi_from_pdf',
                   return_value='10.1000/main'), \
             patch('modules.Forward_Reference.get_paper_details',
                   return_value=('A', 2020, 5, [], 'Main')), \
             patch('modules.Forward_Reference.get_forward_citations',
                   return_value=citing_papers):
            G, _, _ = build_forward_network(sample_pdf_with_doi)

        # The generated node id should start with 'ref_'
        ref_nodes = [n for n in G.nodes if n.startswith('ref_')]
        assert len(ref_nodes) == 1
        assert G.has_edge(ref_nodes[0], '10.1000/main')

    def test_adds_cross_reference_edges(self, sample_pdf_with_doi):
        """If citing paper A references citing paper B (via referenced_ids),
        an edge A → B should be added.

        Note: get_forward_citations renames OpenAlex's 'referenced_works' to
        'referenced_ids', so build_forward_network accesses the
        'referenced_ids' key on each paper dict.
        """
        from modules.Forward_Reference import build_forward_network

        citing_papers = [
            {
                'id': 'W1', 'doi': '10.1/a',
                'display_name': 'A', 'publication_year': 2021, 'cited_by_count': 1,
                'authorships': [{'author': {'display_name': 'X'}}],
                'referenced_ids': ['W2'],  # A references B (note: 'referenced_ids' not 'referenced_works')
            },
            {
                'id': 'W2', 'doi': '10.1/b',
                'display_name': 'B', 'publication_year': 2020, 'cited_by_count': 2,
                'authorships': [{'author': {'display_name': 'Y'}}],
                'referenced_ids': [],
            },
        ]

        with patch('modules.Forward_Reference.extract_doi_from_pdf',
                   return_value='10.1000/main'), \
             patch('modules.Forward_Reference.get_paper_details',
                   return_value=('A', 2020, 50, [], 'Main')), \
             patch('modules.Forward_Reference.get_forward_citations',
                   return_value=citing_papers):
            G, _, _ = build_forward_network(sample_pdf_with_doi)

        # Cross-reference edge: a → b (in addition to a → main and b → main)
        assert G.has_edge('10.1/a', '10.1/b')

    def test_suggestions_have_source_field(self, sample_pdf_with_doi):
        """Each suggestion should have a 'source' field."""
        from modules.Forward_Reference import build_forward_network
        from datetime import datetime

        current_year = datetime.now().year
        citing_papers = [
            {
                'id': f'W{i}', 'doi': f'10.1/c{i}',
                'display_name': f'Paper {i}',
                'publication_year': current_year,  # recent
                'cited_by_count': i * 5,
                'authorships': [{'author': {'display_name': f'Author{i}'}}],
                'referenced_works': [],
            }
            for i in range(1, 11)
        ]

        with patch('modules.Forward_Reference.extract_doi_from_pdf',
                   return_value='10.1000/main'), \
             patch('modules.Forward_Reference.get_paper_details',
                   return_value=('A', 2020, 50, [], 'Main')), \
             patch('modules.Forward_Reference.get_forward_citations',
                   return_value=citing_papers):
            _, suggestions, _ = build_forward_network(sample_pdf_with_doi)

        for s in suggestions:
            assert 'source' in s
            assert s['source'] in (
                'Recent High Impact Citing Paper',
                'High Local Citation Paper',
            )


# =============================================================================
# routes/network.py — _get_extension + validate_file_upload
# =============================================================================

class TestGetExtension:
    """Tests for the _get_extension helper."""

    def test_extracts_pdf_extension(self):
        from routes.network import _get_extension
        assert _get_extension('paper.pdf') == 'pdf'

    def test_extracts_xlsx_extension(self):
        from routes.network import _get_extension
        assert _get_extension('data.xlsx') == 'xlsx'

    def test_handles_uppercase_extension(self):
        from routes.network import _get_extension
        assert _get_extension('Paper.PDF') == 'pdf'

    def test_handles_no_extension(self):
        from routes.network import _get_extension
        assert _get_extension('noextension') == ''

    def test_handles_multiple_dots(self):
        from routes.network import _get_extension
        assert _get_extension('my.paper.final.pdf') == 'pdf'


class TestValidateFileUpload:
    """Tests for validate_file_upload()."""

    def test_rejects_no_file(self):
        from routes.network import validate_file_upload
        valid, error = validate_file_upload(None, 'forward')
        assert valid is False
        assert 'No file' in error

    def test_rejects_empty_filename(self):
        from routes.network import validate_file_upload
        file = MagicMock()
        file.filename = ''
        valid, error = validate_file_upload(file, 'forward')
        assert valid is False
        assert 'No file selected' in error

    def test_rejects_pdf_for_cross_analysis(self):
        """Cross-reference analysis requires Excel, not PDF."""
        from routes.network import validate_file_upload
        file = MagicMock()
        file.filename = 'dois.pdf'
        valid, error = validate_file_upload(file, 'cross')
        assert valid is False
        assert 'Excel' in error

    def test_rejects_excel_for_forward_analysis(self):
        """Forward analysis requires PDF, not Excel."""
        from routes.network import validate_file_upload
        file = MagicMock()
        file.filename = 'paper.xlsx'
        valid, error = validate_file_upload(file, 'forward')
        assert valid is False
        assert 'PDF' in error

    def test_accepts_pdf_for_forward_analysis(self):
        from routes.network import validate_file_upload
        file = MagicMock()
        file.filename = 'paper.pdf'
        valid, error = validate_file_upload(file, 'forward')
        assert valid is True
        assert error is None

    def test_accepts_pdf_for_backward_analysis(self):
        from routes.network import validate_file_upload
        file = MagicMock()
        file.filename = 'paper.pdf'
        valid, error = validate_file_upload(file, 'backward')
        assert valid is True

    def test_accepts_xlsx_for_cross_analysis(self):
        from routes.network import validate_file_upload
        file = MagicMock()
        file.filename = 'dois.xlsx'
        valid, error = validate_file_upload(file, 'cross')
        assert valid is True

    def test_accepts_xls_for_cross_analysis(self):
        from routes.network import validate_file_upload
        file = MagicMock()
        file.filename = 'dois.xls'
        valid, error = validate_file_upload(file, 'cross')
        assert valid is True

    def test_accepts_csv_for_cross_analysis(self):
        from routes.network import validate_file_upload
        file = MagicMock()
        file.filename = 'dois.csv'
        valid, error = validate_file_upload(file, 'cross')
        assert valid is True


# =============================================================================
# routes/network.py — POST /api/network endpoint
# =============================================================================

class TestNetworkEndpoint:
    """Integration tests for POST /api/network."""

    def test_rejects_request_without_file(self, client):
        rv = client.post('/api/network', data={})
        assert rv.status_code == 400
        data = rv.get_json()
        assert data['success'] is False
        assert 'No file' in data['error']

    def test_rejects_invalid_analysis_type(self, client):
        """An invalid 'type' form field should be rejected."""
        from io import BytesIO
        rv = client.post('/api/network', data={
            'file': (BytesIO(b'fake'), 'paper.pdf'),
            'type': 'invalid_type',
        }, content_type='multipart/form-data')
        assert rv.status_code == 400
        data = rv.get_json()
        assert data['success'] is False
        assert 'Invalid analysis type' in data['error']

    def test_rejects_pdf_for_cross_type(self, client):
        """A PDF file with type=cross should be rejected."""
        from io import BytesIO
        rv = client.post('/api/network', data={
            'file': (BytesIO(b'fake'), 'paper.pdf'),
            'type': 'cross',
        }, content_type='multipart/form-data')
        assert rv.status_code == 400
        data = rv.get_json()
        assert 'Excel' in data['error']

    def test_rejects_excel_for_forward_type(self, client):
        """An Excel file with type=forward should be rejected."""
        from io import BytesIO
        rv = client.post('/api/network', data={
            'file': (BytesIO(b'fake'), 'paper.xlsx'),
            'type': 'forward',
        }, content_type='multipart/form-data')
        assert rv.status_code == 400
        data = rv.get_json()
        assert 'PDF' in data['error']

    def test_forward_analysis_returns_graph(self, client):
        """A successful forward analysis should return nodes, edges, suggestions,
        all_papers, and stats."""
        from io import BytesIO
        import networkx as nx

        mock_G = nx.DiGraph()
        mock_G.add_node('10.1/main', is_main=True, author='A', year=2020,
                        citations=50, title='Main', type='main')
        mock_G.add_node('10.1/citing', is_main=False, author='B', year=2021,
                        citations=5, title='Citing', type='citing')
        mock_G.add_edge('10.1/citing', '10.1/main')
        mock_suggestions = [{'doi': '10.1/citing', 'title': 'Citing', 'source': 'Recent'}]
        mock_all_papers = [
            {'Number': 1, 'DOI': '10.1/main', 'Title': 'Main', 'Local Citation Count': 1},
            {'Number': 2, 'DOI': '10.1/citing', 'Title': 'Citing', 'Local Citation Count': 0},
        ]

        with patch('modules.Forward_Reference.build_forward_network',
                   return_value=(mock_G, mock_suggestions, mock_all_papers)):
            rv = client.post('/api/network', data={
                'file': (BytesIO(b'%PDF-1.4 fake'), 'paper.pdf'),
                'type': 'forward',
            }, content_type='multipart/form-data')

        assert rv.status_code == 200
        data = rv.get_json()
        assert data['success'] is True
        assert 'elements' in data
        assert 'nodes' in data['elements']
        assert 'edges' in data['elements']
        assert len(data['elements']['nodes']) == 2
        assert len(data['elements']['edges']) == 1
        assert data['stats']['nodes'] == 2
        assert data['stats']['edges'] == 1
        assert len(data['suggestions']) == 1
        assert len(data['all_papers']) == 2

    def test_backward_analysis_returns_graph(self, client):
        """A successful backward analysis should return a graph."""
        from io import BytesIO
        import networkx as nx

        mock_G = nx.DiGraph()
        mock_G.add_node('10.1/main', is_main=True, author='A', year=2020,
                        citations=50, title='Main')
        mock_G.add_node('10.1/ref', is_main=False, author='B', year=2019,
                        citations=10, title='Ref')
        mock_G.add_edge('10.1/main', '10.1/ref')

        with patch('modules.Backward_Reference.build_reference_network',
                   return_value=(mock_G, [], [])):
            rv = client.post('/api/network', data={
                'file': (BytesIO(b'%PDF-1.4 fake'), 'paper.pdf'),
                'type': 'backward',
            }, content_type='multipart/form-data')

        assert rv.status_code == 200
        data = rv.get_json()
        assert data['success'] is True
        assert data['stats']['nodes'] == 2
        assert data['stats']['edges'] == 1

    def test_cross_analysis_returns_graph(self, client):
        """A successful cross analysis should return a graph."""
        from io import BytesIO
        import networkx as nx

        mock_G = nx.DiGraph()
        mock_G.add_node('10.1/a', author='A', year=2020, citations=10, title='A')
        mock_G.add_node('10.1/b', author='B', year=2019, citations=20, title='B')
        mock_G.add_edge('10.1/a', '10.1/b')

        with patch('modules.Cross_Reference.build_cross_reference_network',
                   return_value=mock_G):
            rv = client.post('/api/network', data={
                'file': (BytesIO(b'fake excel'), 'dois.xlsx'),
                'type': 'cross',
            }, content_type='multipart/form-data')

        assert rv.status_code == 200
        data = rv.get_json()
        assert data['success'] is True
        assert data['stats']['nodes'] == 2
        assert data['stats']['edges'] == 1

    def test_returns_400_when_network_build_fails(self, client):
        """If the build function returns None (no network), return 400."""
        from io import BytesIO

        with patch('modules.Forward_Reference.build_forward_network',
                   return_value=(None, [], [])):
            rv = client.post('/api/network', data={
                'file': (BytesIO(b'%PDF-1.4 fake'), 'paper.pdf'),
                'type': 'forward',
            }, content_type='multipart/form-data')

        assert rv.status_code == 400
        data = rv.get_json()
        assert data['success'] is False

    def test_returns_400_when_network_has_too_few_nodes(self, client):
        """If the built network has < 2 nodes, return 400."""
        from io import BytesIO
        import networkx as nx

        mock_G = nx.DiGraph()
        mock_G.add_node('10.1/main', is_main=True)  # only 1 node

        with patch('modules.Forward_Reference.build_forward_network',
                   return_value=(mock_G, [], [])):
            rv = client.post('/api/network', data={
                'file': (BytesIO(b'%PDF-1.4 fake'), 'paper.pdf'),
                'type': 'forward',
            }, content_type='multipart/form-data')

        assert rv.status_code == 400

    def test_is_main_serialized_as_string(self, client):
        """The is_main boolean should be serialized as 'True'/'False' string."""
        from io import BytesIO
        import networkx as nx

        mock_G = nx.DiGraph()
        mock_G.add_node('10.1/main', is_main=True, author='A', year=2020,
                        citations=10, title='Main')
        mock_G.add_node('10.1/other', is_main=False, author='B', year=2019,
                        citations=5, title='Other')
        mock_G.add_edge('10.1/main', '10.1/other')

        with patch('modules.Forward_Reference.build_forward_network',
                   return_value=(mock_G, [], [])):
            rv = client.post('/api/network', data={
                'file': (BytesIO(b'%PDF-1.4 fake'), 'paper.pdf'),
                'type': 'forward',
            }, content_type='multipart/form-data')

        data = rv.get_json()
        for node in data['elements']['nodes']:
            if node['id'] == '10.1/main':
                assert node['is_main'] == 'True'
            else:
                assert node['is_main'] == 'False'

    def test_temp_file_cleaned_up_after_request(self, client):
        """The uploaded temp file should be deleted after the request completes."""
        from io import BytesIO
        import networkx as nx

        mock_G = nx.DiGraph()
        mock_G.add_node('10.1/a', is_main=True, author='A', year=2020,
                        citations=10, title='A')
        mock_G.add_node('10.1/b', is_main=False, author='B', year=2019,
                        citations=5, title='B')
        mock_G.add_edge('10.1/a', '10.1/b')

        saved_paths = []
        original_save = None

        with patch('modules.Forward_Reference.build_forward_network',
                   return_value=(mock_G, [], [])):
            # Capture the file path that gets saved
            import werkzeug.datastructures
            from routes.network import config

            # We can't easily intercept the save path, but we can check
            # that no temp files are left in the upload folder after the request
            upload_folder = config.UPLOAD_FOLDER
            rv = client.post('/api/network', data={
                'file': (BytesIO(b'%PDF-1.4 fake'), 'paper.pdf'),
                'type': 'forward',
            }, content_type='multipart/form-data')

            assert rv.status_code == 200
            # Check that no .pdf files remain in the upload folder
            import os
            remaining = [f for f in os.listdir(upload_folder) if f.endswith('.pdf')]
            assert len(remaining) == 0, f"Temp files not cleaned up: {remaining}"


# =============================================================================
# Pytest fixtures (Flask app + client)
# =============================================================================

@pytest.fixture
def app():
    from app import app
    app.config['TESTING'] = True
    yield app


@pytest.fixture
def client(app):
    return app.test_client()
