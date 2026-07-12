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
