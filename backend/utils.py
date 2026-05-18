import re
import hashlib
import json
from typing import Optional, Dict, Any, List, Union
from datetime import datetime


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


def validate_api_key(api_key: str, provider: str) -> bool:
    if not api_key:
        return False
    
    invalid_patterns = [
        'your_', 'placeholder', 'example', 'xxx', 'test_key',
        'sk-your', 'api_key_here', 'key_here', '_here',
        'replace_', 'insert_', 'change_'
    ]
    
    key_lower = api_key.lower()
    for pattern in invalid_patterns:
        if pattern in key_lower:
            return False
    
    if len(api_key) < 10:
        return False
    
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
    key_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True, default=str)
    return hashlib.md5(key_data.encode()).hexdigest()


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def parse_year(year_value: Any) -> Optional[int]:
    if year_value is None:
        return None
    
    if isinstance(year_value, int):
        return year_value if 1900 <= year_value <= datetime.now().year + 1 else None
    
    if isinstance(year_value, str):
        try:
            year = int(year_value)
            return year if 1900 <= year <= datetime.now().year + 1 else None
        except ValueError:
            match = re.search(r'\b(19|20)\d{2}\b', year_value)
            if match:
                return int(match.group())
    
    return None


def safe_json_loads(text: str, default: Any = None) -> Any:
    if not text:
        return default
    
    try:
        text = text.strip()
        
        if text.startswith('```'):
            lines = text.split('\n')
            text = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
        
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def format_timestamp(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat() + 'Z'


def calculate_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def parse_smiles(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    
    valid_pattern = r'^[A-Za-z0-9@\[\]\(\)\{\}=#\$:\.\/\\+\-\*]+$'
    
    smiles = smiles.strip()
    if re.match(valid_pattern, smiles):
        return smiles
    
    return None


def is_valid_cas_number(cas: str) -> bool:
    if not cas:
        return False
    
    cas_pattern = r'^\d{2,7}-\d{2}-\d$'
    return bool(re.match(cas_pattern, cas))
