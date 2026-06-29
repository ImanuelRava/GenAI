import os
from dataclasses import dataclass, field
from typing import List, Set, Optional
from pathlib import Path

@dataclass
class Config:
    # core/config.py is nested two levels inside the backend package:
    #   <project_root>/backend/core/config.py
    # So we go parent.parent.parent to reach the project root,
    # and parent.parent to reach the backend directory.
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    BACKEND_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    UPLOAD_FOLDER: str = ""
    ALLOWED_EXTENSIONS: Set[str] = field(default_factory=lambda: {'pdf', 'xlsx', 'xls', 'csv'})
    MAX_FILE_SIZE: int = 16 * 1024 * 1024
    NETWORK_PDF_EXTENSIONS: Set[str] = field(default_factory=lambda: {'pdf'})
    NETWORK_EXCEL_EXTENSIONS: Set[str] = field(default_factory=lambda: {'xlsx', 'xls', 'csv'})
    MAX_PDF_PAGES: int = 50

    CORS_ORIGINS: List[str] = field(default_factory=list)

    RATE_LIMIT_DEFAULT: str = "200 per day;50 per hour"
    RATE_LIMIT_NETWORK: str = "10 per minute"
    RATE_LIMIT_LLM: str = "20 per minute"

    LLM_TIMEOUT: int = 60
    LLM_MAX_TOKENS: int = 2000
    LLM_TEMPERATURE: float = 0.7
    MAX_PROMPT_LENGTH: int = 2000
    # Larger limit for the /api/extract text-extraction endpoint, which
    # intentionally accepts longer passages than the chat endpoints.
    MAX_EXTRACTION_TEXT_LENGTH: int = 15000

    CACHE_ENABLED: bool = True
    CACHE_TTL_PAPER: int = 86400
    CACHE_TTL_LLM: int = 3600
    CACHE_MAX_SIZE: int = 500

    SECRET_KEY: str = ""
    SESSION_COOKIE_SECURE: bool = True

    def __post_init__(self):
        self.UPLOAD_FOLDER = str(self.BACKEND_DIR / 'uploads')
        # Ensure upload directory exists
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)

        cors_default = 'http://localhost:5000,http://127.0.0.1:5000,http://localhost:3000'
        cors_env = os.environ.get('CORS_ORIGINS', cors_default)
        self.CORS_ORIGINS = [origin.strip() for origin in cors_env.split(',') if origin.strip()]

        self.SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(32).hex())

        redis_url = os.environ.get('REDIS_URL')
        if redis_url:
            self.CACHE_TYPE = 'redis'
            self.CACHE_REDIS_URL = redis_url
        else:
            self.CACHE_TYPE = 'memory'

    @property
    def static_folder(self) -> str:
        return str(self.BASE_DIR)

    @property
    def database_path(self) -> str:
        db_path = self.BACKEND_DIR / 'genai.db'
        return str(db_path)

config = Config()

LLM_PROVIDER_CONFIG = {
    'groq': {
        'name': 'Groq',
        'url': 'https://console.groq.com',
        'base_url': 'https://api.groq.com/openai/v1',
        'default_model': 'llama-3.3-70b-versatile',
        'free_tier': True
    },
    'gemini': {
        'name': 'Google Gemini',
        'url': 'https://ai.google.dev',
        'default_model': 'gemini-2.0-flash',
        'free_tier': True
    },
    'deepseek': {
        'name': 'DeepSeek',
        'url': 'https://platform.deepseek.com',
        'base_url': 'https://api.deepseek.com/v1',
        'default_model': 'deepseek-chat',
        'free_tier': False
    },
    'openai': {
        'name': 'OpenAI',
        'url': 'https://platform.openai.com',
        'base_url': 'https://api.openai.com/v1',
        'default_model': 'gpt-4o-mini',
        'free_tier': False
    },
    'anthropic': {
        'name': 'Anthropic',
        'url': 'https://console.anthropic.com',
        'base_url': 'https://api.anthropic.com/v1',
        'default_model': 'claude-3-haiku-20240307',
        'free_tier': False
    },
    'huggingface': {
        'name': 'Hugging Face',
        'url': 'https://huggingface.co/settings/tokens',
        'default_model': 'meta-llama/Llama-3.2-3B-Instruct',
        'free_tier': True
    },
    'openrouter': {
        'name': 'OpenRouter',
        'url': 'https://openrouter.ai/keys',
        'base_url': 'https://openrouter.ai/api/v1',
        'default_model': 'meta-llama/llama-3-8b-instruct:free',
        'free_tier': True
    },
    'ollama': {
        'name': 'Ollama (Local)',
        'url': 'https://ollama.ai',
        'base_url': 'http://localhost:11434',
        'default_model': 'llama3',
        'free_tier': True
    }
}
