"""
Utility Functions for GenAI Research Platform
Includes input sanitization, validation, and helper functions
"""

import re
import hashlib
import json
from typing import Optional, Dict, Any, List, Union
from datetime import datetime


def sanitize_input(text: str, max_length: int = 2000) -> str:
    """
    Sanitize user input for LLM prompts to prevent injection attacks.
    
    Args:
        text: Raw user input string
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string safe for LLM prompts
    """
    if not text:
        return ""
    
    # Truncate to max length
    text = text[:max_length]
    
    # Remove potential prompt injection patterns
    # Remove common injection patterns but preserve chemistry content
    injection_patterns = [
        r'ignore\s+(all\s+)?previous\s+instructions?',
        r'system\s*:\s*',
        r'<\|.*?\|>',  # Special tokens
        r'\[SYSTEM\]',
        r'\[INST\]',
        r'<<<.*?>>>',
    ]
    
    for pattern in injection_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    return text.strip()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Remove path separators and null bytes
    filename = filename.replace('/', '_').replace('\\', '_').replace('\x00', '')
    
    # Remove potentially dangerous characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:250] + ('.' + ext if ext else '')
    
    return filename


def validate_doi(doi: str) -> bool:
    """
    Validate DOI format.
    
    Args:
        doi: DOI string to validate
        
    Returns:
        True if valid DOI format
    """
    if not doi:
        return False
    
    # DOI pattern: 10.xxxx/xxxxx
    doi_pattern = r'^10\.\d{4,}/[^\s]+$'
    return bool(re.match(doi_pattern, doi))


def validate_api_key(api_key: str, provider: str) -> bool:
    """
    Validate API key format for a given provider.
    
    Args:
        api_key: API key string
        provider: Provider name
        
    Returns:
        True if key appears valid
    """
    if not api_key:
        return False
    
    # Check for placeholder patterns
    invalid_patterns = [
        'your_', 'placeholder', 'example', 'xxx', 'test_key',
        'sk-your', 'api_key_here', 'key_here', '_here',
        'replace_', 'insert_', 'change_'
    ]
    
    key_lower = api_key.lower()
    for pattern in invalid_patterns:
        if pattern in key_lower:
            return False
    
    # Minimum length check (real API keys are longer)
    if len(api_key) < 10:
        return False
    
    # Provider-specific validation
    provider_rules = {
        'openai': lambda k: k.startswith('sk-'),
        'anthropic': lambda k: k.startswith('sk-ant-'),
        'groq': lambda k: k.startswith('gsk_') or k.startswith('sk-'),
        'deepseek': lambda k: k.startswith('sk-'),
    }
    
    if provider in provider_rules:
        return provider_rules[provider](api_key)
    
    return True


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a consistent cache key from arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        MD5 hash string for caching
    """
    key_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True, default=str)
    return hashlib.md5(key_data.encode()).hexdigest()


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """
    Truncate text to a maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to append when truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def parse_year(year_value: Any) -> Optional[int]:
    """
    Parse year from various input formats.
    
    Args:
        year_value: Year value (int, str, or other)
        
    Returns:
        Integer year or None
    """
    if year_value is None:
        return None
    
    if isinstance(year_value, int):
        return year_value if 1900 <= year_value <= datetime.now().year + 1 else None
    
    if isinstance(year_value, str):
        try:
            year = int(year_value)
            return year if 1900 <= year <= datetime.now().year + 1 else None
        except ValueError:
            # Try to extract 4-digit year from string
            match = re.search(r'\b(19|20)\d{2}\b', year_value)
            if match:
                return int(match.group())
    
    return None


def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON with fallback.
    
    Args:
        text: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    if not text:
        return default
    
    try:
        # Clean up common JSON issues
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith('```'):
            lines = text.split('\n')
            text = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
        
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format datetime as ISO timestamp.
    
    Args:
        dt: Datetime object (defaults to now)
        
    Returns:
        ISO format timestamp string
    """
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat() + 'Z'


def calculate_hash(text: str) -> str:
    """
    Calculate SHA256 hash of text.
    
    Args:
        text: Text to hash
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(text.encode()).hexdigest()


# Chemistry-specific utilities

def parse_smiles(smiles: str) -> Optional[str]:
    """
    Validate and clean SMILES string.
    
    Args:
        smiles: SMILES notation string
        
    Returns:
        Cleaned SMILES or None if invalid
    """
    if not smiles:
        return None
    
    # Basic validation - SMILES should only contain certain characters
    valid_pattern = r'^[A-Za-z0-9@\[\]\(\)\{\}=#\$:\.\/\\+\-\*]+$'
    
    smiles = smiles.strip()
    if re.match(valid_pattern, smiles):
        return smiles
    
    return None


def is_valid_cas_number(cas: str) -> bool:
    """
    Validate CAS registry number format.
    
    Args:
        cas: CAS number string
        
    Returns:
        True if valid CAS format
    """
    if not cas:
        return False
    
    # CAS format: XXXXXXX-XX-X
    cas_pattern = r'^\d{2,7}-\d{2}-\d$'
    return bool(re.match(cas_pattern, cas))
