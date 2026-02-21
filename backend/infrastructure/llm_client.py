from typing import Dict, Any, Optional
import json
from openai import AsyncOpenAI
from langchain_openai import ChatOpenAI
from backend.core.config import settings
from backend.infrastructure.bailian_llm import BailianChatOpenAI

class LLMClient:
    def __init__(self):
        # Initialize OpenAI client for legacy support
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE
        )
        
    def get_chat_model(self, temperature: float = 0.7, model_name: Optional[str] = None) -> ChatOpenAI:
        """Get LangChain Chat Model"""
        base_url = settings.OPENAI_API_BASE
        
        # Check if we should use the Bailian wrapper
        if "aliyuncs.com" in base_url or "bailian" in base_url:
            return BailianChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=base_url,
                model_name=model_name or settings.MODEL_NAME,
                temperature=temperature
            )
            
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE,
            model_name=model_name or settings.MODEL_NAME,
            temperature=temperature
        )

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        response_format: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None
    ) -> str:
        """Legacy completion method (kept for backward compatibility)"""
        messages = [{"role": "user", "content": prompt}]
        
        kwargs = {
            "model": model or settings.MODEL_NAME,
            "messages": messages,
            "temperature": temperature,
        }
        
        if response_format:
            kwargs["response_format"] = response_format

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
