from .config import (
    TEXT_CHUNK_SIZE,
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    RETRY_DELAY,
    EXTRACTION_TEMPERATURE,
    EXTRACTION_SEED,
    CHUNK_OVERLAP,
    CACHE_DIR,
    RGROUP_SMILES_REFERENCE,
)
from .prompts import (
    SYSTEM_PROMPT_CHEMICAL_ENTITIES,
    SYSTEM_PROMPT_REACTION_SCHEME,
    SYSTEM_PROMPT_TABLE_EXTRACTION,
    SYSTEM_PROMPT_VISION,
    SYSTEM_PROMPT_FIGURE_ANALYSIS,
    SYSTEM_PROMPT_COMPREHENSIVE,
)
from .pdf_processor import (
    extract_text_from_pdf,
    pdf_to_images,
    extract_figures_from_pdf,
    render_scheme_pages,
    detect_scheme_pages,
    extract_all_visual_content,
)
from .llm_providers import (
    call_vision_llm,
    call_vision_llm_async,
    _retry_on_failure,
    _retry_on_failure_async,
)
from .json_utils import (
    _parse_json_response,
)
from .smiles_utils import (
    assemble_rgroup_smiles,
    assemble_rgroup_reactions,
)
from .reaction_formatter import (
    format_reaction_schemes,
    format_reaction_schemes_simple,
)
from .standalone import (
    call_text_llm,
    call_text_llm_async,
    call_text_llm_chunked,
    call_text_llm_chunked_async,
)
from .extractor import (
    ChemExtractAI,
    extract_chemical_data_from_pdf,
    extract_chemical_data_from_pdf_async,
)

__all__ = [
    # Config
    "TEXT_CHUNK_SIZE", "MAX_OUTPUT_TOKENS", "MAX_RETRIES", "RETRY_DELAY",
    "EXTRACTION_TEMPERATURE", "EXTRACTION_SEED", "CHUNK_OVERLAP", "CACHE_DIR",
    "RGROUP_SMILES_REFERENCE",
    # Prompts
    "SYSTEM_PROMPT_CHEMICAL_ENTITIES", "SYSTEM_PROMPT_REACTION_SCHEME",
    "SYSTEM_PROMPT_TABLE_EXTRACTION", "SYSTEM_PROMPT_VISION",
    "SYSTEM_PROMPT_FIGURE_ANALYSIS", "SYSTEM_PROMPT_COMPREHENSIVE",
    # PDF processing
    "extract_text_from_pdf", "pdf_to_images",
    "extract_figures_from_pdf", "render_scheme_pages",
    "detect_scheme_pages", "extract_all_visual_content",
    # LLM calls
    "call_vision_llm", "call_vision_llm_async",
    "call_text_llm", "call_text_llm_async",
    "call_text_llm_chunked", "call_text_llm_chunked_async",
    # Retry helpers (re-exported for mermaid_integration.py compatibility)
    "_retry_on_failure", "_retry_on_failure_async",
    # JSON parsing (re-exported for compatibility)
    "_parse_json_response",
    # SMILES utilities
    "assemble_rgroup_smiles", "assemble_rgroup_reactions",
    # Formatting
    "format_reaction_schemes", "format_reaction_schemes_simple",
    # Main class
    "ChemExtractAI",
    "extract_chemical_data_from_pdf", "extract_chemical_data_from_pdf_async",
]
