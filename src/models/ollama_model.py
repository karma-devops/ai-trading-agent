"""
🌙 Moon Dev's Ollama Model Integration
Built with love by Moon Dev 🚀

This module provides integration with locally running Ollama models.
"""

import os
import requests
import json
from termcolor import cprint
from .base_model import BaseModel, ModelResponse

class OllamaModel(BaseModel):
    """Implementation for Ollama models (local or OpenAI-compatible cloud endpoint)."""

    DEFAULT_BASE_URL = os.getenv("AI_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api"))

    def __init__(self, api_key=None, model_name="kimi-k2.7-code", base_url=None):
        """Initialize Ollama model

        Args:
            api_key: API key (for OpenAI-compatible cloud endpoints like ollama.com/v1)
            model_name: Name of the model to use
            base_url: Custom API endpoint. If URL contains /v1, uses OpenAI-compatible client.
        """
        self._api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.model_name = model_name
        self._is_connected = False
        self._connection_error = None
        self._available_models = []
        self._use_openai_compat = False
        self._openai_client = None

        super().__init__(api_key="LOCAL_OLLAMA")
        self.initialize_client()

    def initialize_client(self):
        """Initialize the Ollama client connection.
        Detects if base_url is OpenAI-compatible (/v1) and uses OpenAI client."""
        self._is_connected = False
        self._connection_error = None
        self._use_openai_compat = False
        self._openai_client = None

        # If base_url contains /v1, use OpenAI-compatible client
        if self.base_url and '/v1' in self.base_url:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(
                    api_key=self._api_key if self._api_key else "ollama",
                    base_url=self.base_url
                )
                self._is_connected = True
                self._use_openai_compat = True
                cprint(f"✨ Connected to Ollama via OpenAI-compatible API: {self.base_url}", "green")
                cprint(f"   Model: {self.model_name}", "cyan")
                return
            except Exception as e:
                self._connection_error = f"OpenAI client init failed: {e}"
                cprint(f"⚠️ {self._connection_error}", "yellow")
                return

        # Native Ollama API (localhost:11434)
        try:
            response = requests.get(f"{self.base_url}/tags", timeout=5)
            if response.status_code == 200:
                self._is_connected = True
                cprint(f"✨ Connected to Ollama server at {self.base_url}", "green")

                models = response.json().get("models", [])
                if models:
                    self._available_models = [model["name"] for model in models]
                    cprint(f"📚 {len(self._available_models)} models available locally", "cyan")

                    if self.model_name not in self._available_models:
                        partial_matches = [m for m in self._available_models if self.model_name in m or m in self.model_name]
                        if partial_matches:
                            cprint(f"   Using closest match: {partial_matches[0]}", "cyan")
                            self.model_name = partial_matches[0]
                        else:
                            cprint(f"⚠️ Model '{self.model_name}' not found locally!", "yellow")
            else:
                self._connection_error = f"Ollama API returned status code: {response.status_code}"
                cprint(f"⚠️ {self._connection_error}", "yellow")

        except requests.exceptions.ConnectionError:
            self._connection_error = "Ollama server not running"
            cprint(f"⚠️ Ollama server not running at {self.base_url}", "yellow")
            cprint("   💡 Start with: ollama serve", "cyan")

        except requests.exceptions.Timeout:
            self._connection_error = "Connection timeout"
            cprint("⚠️ Ollama server connection timed out", "yellow")

        except Exception as e:
            self._connection_error = str(e)
            cprint(f"⚠️ Ollama connection error: {str(e)}", "yellow")

    @property
    def model_type(self):
        """Return the type of model"""
        return "ollama"

    def is_available(self):
        """Check if the Ollama server is connected and available"""
        return self._is_connected

    def get_connection_status(self):
        """Get detailed connection status

        Returns:
            dict with keys:
                - connected: bool
                - error: str or None
                - available_models: list of model names
                - base_url: str
        """
        return {
            "connected": self._is_connected,
            "error": self._connection_error,
            "available_models": self._available_models,
            "base_url": self.base_url,
            "current_model": self.model_name
        }

    def reconnect(self):
        """Attempt to reconnect to the Ollama server"""
        cprint("🔄 Attempting to reconnect to Ollama server...", "cyan")
        self.initialize_client()
        return self._is_connected
    
    def generate_response(self, system_prompt, user_content, temperature=0.7, max_tokens=None, **kwargs):
        """Generate a response using Ollama (native or OpenAI-compatible)."""
        if not self._is_connected:
            self.reconnect()
            if not self._is_connected:
                return ModelResponse(
                    content="",
                    raw_response={"error": f"Ollama not available: {self._connection_error}"},
                    model_name=self.model_name,
                    usage=None
                )

        # OpenAI-compatible mode
        if self._use_openai_compat and self._openai_client:
            try:
                response = self._openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens if max_tokens else 4000,
                    stream=False
                )
                raw_content = response.choices[0].message.content or ""

                # Strip reasoning tags
                import re
                filtered = re.sub(r'ILD.*?ILD', '', raw_content, flags=re.DOTALL).strip()
                if 'ILD' in filtered:
                    filtered = filtered.split('ILD')[0].strip()
                final = filtered if filtered else raw_content

                return ModelResponse(
                    content=final,
                    raw_response=response,
                    model_name=self.model_name,
                    usage=getattr(response, 'usage', None)
                )
            except Exception as e:
                cprint(f"❌ Ollama OpenAI-compatible error: {str(e)}", "red")
                return ModelResponse(
                    content="",
                    raw_response={"error": str(e)},
                    model_name=self.model_name,
                    usage=None
                )

        # Native Ollama mode
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            data = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }

            response = requests.post(
                f"{self.base_url}/chat",
                json=data,
                timeout=90
            )

            if response.status_code == 200:
                response_data = response.json()
                raw_content = response_data.get("message", {}).get("content", "")

                import re
                filtered_content = re.sub(r'ILD.*?ILD', '', raw_content, flags=re.DOTALL).strip()
                if 'ILD' in filtered_content:
                    filtered_content = filtered_content.split('ILD')[0].strip()
                final_content = filtered_content if filtered_content else raw_content

                return ModelResponse(
                    content=final_content,
                    raw_response=response_data,
                    model_name=self.model_name,
                    usage=None
                )
            else:
                cprint(f"❌ Ollama API error: {response.status_code}", "red")
                raise Exception(f"Ollama API error: {response.status_code}")

        except Exception as e:
            cprint(f"❌ Error generating response: {str(e)}", "red")
            return ModelResponse(
                content="",
                raw_response={"error": str(e)},
                model_name=self.model_name,
                usage=None
            )
    
    def __str__(self):
        return f"OllamaModel(model={self.model_name})"

    def get_model_parameters(self, model_name=None):
        """Get the parameter count for a specific model

        Args:
            model_name: Name of the model to check (defaults to self.model_name)

        Returns:
            String with parameter count (e.g., "7B", "13B") or None if not available
        """
        if model_name is None:
            model_name = self.model_name

        try:
            # Check AVAILABLE_MODELS dict for parameter info
            if model_name in self.AVAILABLE_MODELS:
                return self.AVAILABLE_MODELS[model_name].get("parameters", "Unknown")

            return "Unknown"
        except Exception as e:
            cprint(f"❌ Error getting model parameters: {str(e)}", "red")
            return None 