import re
from typing import Optional


def sanitize_input(text: str, max_length: int = 2000) -> str:
    if not text:
        return ""
    
    text = text[:max_length]
    
    injection_patterns = [
        r'ignore\s+(all\s+)?previous\s+instructions?',
        r'system\s*:\s*',
        r'<\|.*?\|>',
        r'\[SYSTEM\]',
        r'\[INST\]',
        r'<<<.*?>>>',
    ]
    
    for pattern in injection_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    return text.strip()


def sanitize_filename(filename: str) -> str:
    filename = filename.replace('/', '_').replace('\\', '_').replace('\x00', '')

    filename = re.sub(r'[^\w\-_\.]', '_', filename)

    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:250] + ('.' + ext if ext else '')
    
    return filename


def validate_doi(doi: str) -> bool:
    if not doi:
        return False
    
    doi_pattern = r'^10\.\d{4,}/[^\s]+$'
    return bool(re.match(doi_pattern, doi))
