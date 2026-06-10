import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _parse_json_response(content: str) -> Optional[Dict]:
    if not content:
        return None

    content = content.strip()
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        pass

    depth = 0
    start = -1
    best_json = None

    for i, char in enumerate(content):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    candidate = json.loads(content[start:i+1])
                    if best_json is None or len(candidate) > len(best_json):
                        best_json = candidate
                except (json.JSONDecodeError, ValueError):
                    pass

    if best_json:
        return best_json

    try:
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        if open_braces > 0 or open_brackets > 0:
            fixed = content + '}' * max(0, open_braces) + ']' * max(0, open_brackets)
            return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    logger.warning(f"[ChemExtract] JSON parse failed for content of length {len(content)}")
    return None
