"""
Generic OpenAI-compatible model provider.
Supports OpenRouter, HuggingFace Inference API, Ollama.com /v1, and any other
OpenAI-compatible endpoint via base_url.
"""

import os
import time
import re
from openai import OpenAI
from termcolor import cprint
from .base_model import BaseModel, ModelResponse


class GenericOpenAIModel(BaseModel):
    """Implementation for any OpenAI-compatible API endpoint."""

    def __init__(self, api_key: str, model_name: str = "", base_url: str = "", **kwargs):
        self.model_name = model_name
        self.base_url = base_url or kwargs.get('base_url', '')
        self.max_tokens = kwargs.get('max_tokens', 2000)
        super().__init__(api_key, **kwargs)

    def initialize_client(self, **kwargs) -> None:
        """Initialize the OpenAI-compatible client."""
        if not self.api_key:
            raise ValueError("API key is required for generic OpenAI-compatible provider")
        if not self.base_url:
            raise ValueError("base_url is required for generic OpenAI-compatible provider")
        if not self.model_name:
            raise ValueError("model_name is required for generic OpenAI-compatible provider")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        cprint(f"✨ Generic OpenAI-compatible client initialized: {self.model_name}", "green")
        cprint(f"   Base URL: {self.base_url}", "cyan")

    def generate_response(self, system_prompt, user_content, temperature=0.7, max_tokens=None, **kwargs):
        """Generate response from the OpenAI-compatible endpoint."""
        if not self.client:
            cprint("❌ Generic OpenAI client not initialized", "red")
            return None

        try:
            # Force unique request every time to avoid caching
            timestamp = int(time.time() * 1000)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=max_tokens if max_tokens else self.max_tokens,
                stream=False
            )

            raw_content = response.choices[0].message.content or ""

            # Strip reasoning tags if present
            filtered = re.sub(r'\s*<think>.*?</think>\s*', '', raw_content, flags=re.DOTALL)
            if '<think>' in filtered:
                filtered = filtered.split('<think>')[0].strip()
            final = filtered if filtered else raw_content

            return ModelResponse(
                content=final,
                raw_response=response,
                model_name=self.model_name,
                usage=response.usage
            )

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str:
                cprint(f"⚠️  Rate limited: {error_str[:80]}", "yellow")
            elif "402" in error_str or "insufficient" in error_str:
                cprint(f"⚠️  Credits insufficient: {error_str[:80]}", "yellow")
            elif "503" in error_str:
                raise e
            else:
                cprint(f"❌ Provider error: {error_str[:120]}", "red")
            return ModelResponse(
                content="",
                raw_response={"error": error_str},
                model_name=self.model_name,
                usage=None
            )

    def is_available(self) -> bool:
        return self.client is not None

    @property
    def model_type(self) -> str:
        return "generic_openai"
