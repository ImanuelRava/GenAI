"""
Pure Python LLM Providers for PythonAnywhere
Supports DeepSeek, OpenAI, Anthropic, Ollama, Groq (free tier), and Hugging Face (free)
"""

import os
import json
import requests
from typing import Optional, Dict, Any


# FREE LLM OPTIONS:
# 1. Groq (FREE tier) - https://console.groq.com - Very fast, generous free tier
# 2. Hugging Face (FREE) - https://huggingface.co - Free inference API
# 3. Ollama (100% FREE) - Run locally on your machine - https://ollama.ai
# 4. Google Gemini (FREE tier) - https://ai.google.dev - Free tier available


class BaseLLMProvider:
    """Base class for LLM providers"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
    
    def chat(self, system_prompt: str, user_message: str, **kwargs) -> Optional[str]:
        """Send chat completion request"""
        raise NotImplementedError("Subclasses must implement chat()")
    
    def _make_request(self, headers: Dict, payload: Dict) -> Optional[str]:
        """Make HTTP request to LLM API"""
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                print(f"LLM API Error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("LLM request timed out")
            return None
        except Exception as e:
            print(f"LLM request error: {e}")
            return None


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek API Provider"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY')
        self.base_url = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
        self.model = os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat')
    
    def chat(self, system_prompt: str, user_message: str, 
             temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """Send chat completion request to DeepSeek"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        return self._make_request(headers, payload)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API Provider"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        self.model = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
    
    def chat(self, system_prompt: str, user_message: str,
             temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """Send chat completion request to OpenAI"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        return self._make_request(headers, payload)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API Provider"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.base_url = 'https://api.anthropic.com/v1'
        self.model = os.environ.get('ANTHROPIC_MODEL', 'claude-3-haiku-20240307')
    
    def chat(self, system_prompt: str, user_message: str,
             temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """Send chat completion request to Anthropic"""
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['content'][0]['text']
            else:
                print(f"Anthropic API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Anthropic request error: {e}")
            return None


class OllamaProvider(BaseLLMProvider):
    """Ollama Local LLM Provider (for local development)"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.model = os.environ.get('OLLAMA_MODEL', 'llama3')
    
    def chat(self, system_prompt: str, user_message: str,
             temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """Send chat completion request to Ollama"""
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['message']['content']
            else:
                print(f"Ollama API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Ollama request error: {e}")
            return None


class GroqProvider(BaseLLMProvider):
    """
    Groq API Provider - FREE TIER AVAILABLE!
    
    Get your free API key at: https://console.groq.com
    - Generous free tier with fast inference
    - Models: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768
    - No credit card required for free tier
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('GROQ_API_KEY')
        self.base_url = 'https://api.groq.com/openai/v1'
        self.model = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
    
    def chat(self, system_prompt: str, user_message: str,
             temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """Send chat completion request to Groq"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        return self._make_request(headers, payload)


class HuggingFaceProvider(BaseLLMProvider):
    """
    Hugging Face Inference API Provider - FREE!
    
    Get your free API key at: https://huggingface.co/settings/tokens
    - Completely free for inference
    - Many models available
    - Rate limits apply on free tier
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get('HF_API_KEY') or os.environ.get('HUGGINGFACE_API_KEY')
        self.model = model or os.environ.get('HF_MODEL', 'meta-llama/Llama-3.2-3B-Instruct')
        self.base_url = f'https://api-inference.huggingface.co/models/{self.model}'
    
    def chat(self, system_prompt: str, user_message: str,
             temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """Send chat completion request to Hugging Face"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Format as instruction for instruct models
        formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{user_message}\n<|assistant|"
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get('generated_text', '')
                return data.get('generated_text', '')
            elif response.status_code == 503:
                # Model is loading, wait and retry
                print("Hugging Face: Model is loading, please wait...")
                return None
            else:
                print(f"Hugging Face API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Hugging Face request error: {e}")
            return None


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini API Provider - FREE TIER!
    
    Get your free API key at: https://ai.google.dev
    - Free tier with generous limits
    - Models: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        self.model = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')
    
    def chat(self, system_prompt: str, user_message: str,
             temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """Send chat completion request to Google Gemini"""
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{system_prompt}\n\n{user_message}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['candidates'][0]['content']['parts'][0]['text']
            else:
                print(f"Gemini API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Gemini request error: {e}")
            return None


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter API Provider - Works in Hong Kong!
    
    Get your API key at: https://openrouter.ai
    - Aggregates multiple LLM providers
    - Free credits available
    - Works globally including Hong Kong
    - Models: meta-llama/llama-3-8b-instruct (free), openai/gpt-4o-mini, etc.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        self.base_url = 'https://openrouter.ai/api/v1'
        # Default to free model
        self.model = os.environ.get('OPENROUTER_MODEL', 'meta-llama/llama-3-8b-instruct:free')
    
    def chat(self, system_prompt: str, user_message: str,
             temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """Send chat completion request to OpenRouter"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://genai-research.local",  # Optional, for rankings
            "X-Title": "GenAI Research"  # Optional
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        return self._make_request(headers, payload)


class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    # Priority order for providers that work in Hong Kong
    HK_FRIENDLY_PROVIDERS = ['ollama', 'openrouter', 'huggingface', 'deepseek']
    ALL_PROVIDERS = ['ollama', 'openrouter', 'huggingface', 'deepseek', 'gemini', 'groq', 'openai', 'anthropic']
    
    @staticmethod
    def create(provider: str = 'ollama', **kwargs) -> BaseLLMProvider:
        """Create an LLM provider instance
        
        Default is 'ollama' (100% FREE, works everywhere including Hong Kong)
        For deployed websites, use 'openrouter', 'deepseek', or 'huggingface'
        """
        
        providers = {
            'ollama': OllamaProvider,
            'openrouter': OpenRouterProvider,
            'huggingface': HuggingFaceProvider,
            'hf': HuggingFaceProvider,  # alias
            'deepseek': DeepSeekProvider,
            'gemini': GeminiProvider,
            'groq': GroqProvider,
            'openai': OpenAIProvider,
            'anthropic': AnthropicProvider
        }
        
        if provider.lower() not in providers:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
        
        return providers[provider.lower()](**kwargs)
    
    @staticmethod
    def get_default_provider() -> str:
        """Get the default provider based on available environment variables
        
        Priority order (Hong Kong friendly first):
        1. DeepSeek (DEEPSEEK_API_KEY) - Chinese company, works in HK
        2. OpenRouter (OPENROUTER_API_KEY) - Works in HK, has free models
        3. Hugging Face (HF_API_KEY) - FREE tier, works globally
        4. Ollama (OLLAMA_BASE_URL) - 100% FREE, local only
        5. Gemini (GEMINI_API_KEY) - May not work in HK
        6. Groq (GROQ_API_KEY) - May not work in HK
        7. OpenAI (OPENAI_API_KEY)
        8. Anthropic (ANTHROPIC_API_KEY)
        """
        # Check Hong Kong friendly providers first
        if os.environ.get('DEEPSEEK_API_KEY'):
            return 'deepseek'
        elif os.environ.get('OPENROUTER_API_KEY'):
            return 'openrouter'
        elif os.environ.get('HF_API_KEY') or os.environ.get('HUGGINGFACE_API_KEY'):
            return 'huggingface'
        elif os.environ.get('OLLAMA_BASE_URL'):
            return 'ollama'
        # Then check other providers
        elif os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'):
            return 'gemini'
        elif os.environ.get('GROQ_API_KEY'):
            return 'groq'
        elif os.environ.get('OPENAI_API_KEY'):
            return 'openai'
        elif os.environ.get('ANTHROPIC_API_KEY'):
            return 'anthropic'
        
        return 'deepseek'  # Default to DeepSeek (works in Hong Kong)


def get_llm_response(system_prompt: str, user_message: str, 
                     provider: str = None, **kwargs) -> Optional[str]:
    """
    Get response from LLM using configured provider
    
    Args:
        system_prompt: System message for the LLM
        user_message: User message/query
        provider: LLM provider name (deepseek, openai, anthropic, ollama)
        **kwargs: Additional arguments passed to the provider
    
    Returns:
        LLM response string or None if failed
    """
    if provider is None:
        provider = LLMProviderFactory.get_default_provider()
    
    try:
        llm = LLMProviderFactory.create(provider)
        return llm.chat(system_prompt, user_message, **kwargs)
    except Exception as e:
        print(f"LLM Error: {e}")
        return None


# Convenience functions for common tasks

def generate_knowledge_graph(topic: str, provider: str = None) -> Optional[Dict]:
    """
    Generate a knowledge graph for a given topic using LLM
    
    Args:
        topic: The topic to generate knowledge graph for
        provider: LLM provider to use
    
    Returns:
        Dictionary with nodes and edges, or None if failed
    """
    system_prompt = """You are an expert in transition metal catalysis and chemistry education.
Generate a knowledge graph for the given topic in valid JSON format ONLY (no markdown, no explanation).
Return ONLY a JSON object with this exact structure:
{
    "nodes": [
        {"id": "unique_snake_case_id", "label": "Display Name", "type": "reaction|catalyst|reagent|mechanism|product|ligand|property", "description": "Brief 1-2 sentence description"}
    ],
    "edges": [
        {"source": "node_id", "target": "node_id", "label": "relationship"}
    ]
}
Rules:
- Include 8-15 nodes
- Use snake_case for IDs
- Types must be one of: reaction, catalyst, reagent, mechanism, product, ligand, property
- Make connections educationally meaningful
- Focus on chemical accuracy
- Return ONLY the JSON, no other text"""

    user_message = f"Generate a knowledge graph for: {topic}"
    
    response = get_llm_response(system_prompt, user_message, provider=provider)
    
    if response:
        try:
            # Clean up response
            json_str = response.strip()
            if json_str.startswith('```'):
                lines = json_str.split('\n')
                json_str = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Response was: {response[:500]}...")
    
    return None


def explain_concept(concept: str, context: str = "", provider: str = None) -> Optional[str]:
    """
    Get an explanation for a chemistry concept
    
    Args:
        concept: The concept to explain
        context: Additional context
        provider: LLM provider to use
    
    Returns:
        Explanation string or None if failed
    """
    system_prompt = """You are an expert chemistry educator specializing in transition metal catalysis.
Provide a clear, concise explanation (2-3 sentences) for the given chemistry concept.
Focus on practical understanding and real-world applications.
Keep the explanation accessible to graduate-level chemistry students."""

    user_message = f"Explain {concept} in the context of transition metal catalysis. Context: {context}"
    
    return get_llm_response(system_prompt, user_message, provider=provider)


# Test function
if __name__ == "__main__":
    print("Testing LLM Provider...")
    
    # Test with environment variables
    provider = LLMProviderFactory.get_default_provider()
    print(f"Default provider: {provider}")
    
    # Test a simple query
    response = get_llm_response(
        "You are a helpful chemistry assistant.",
        "What is Suzuki coupling in one sentence?"
    )
    
    if response:
        print(f"Response: {response}")
    else:
        print("No response received. Check your API key configuration.")
