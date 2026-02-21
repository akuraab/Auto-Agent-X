from typing import Any, Dict, List, Optional, AsyncIterator, Iterator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
import logging

logger = logging.getLogger(__name__)

class BailianChatOpenAI(ChatOpenAI):
    """
    Custom wrapper for Aliyun Bailian (DashScope) OpenAI-compatible API.
    Handles non-standard response formats and error messages.
    """
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return super()._generate(messages, stop=stop, **kwargs)
        except Exception as e:
            self._handle_bailian_error(e)
            raise e

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return await super()._agenerate(messages, stop=stop, **kwargs)
        except Exception as e:
            self._handle_bailian_error(e)
            raise e

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        try:
            for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                yield chunk
        except Exception as e:
            self._handle_bailian_error(e)
            raise e

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        try:
            async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
                yield chunk
        except Exception as e:
            self._handle_bailian_error(e)
            raise e

    def _handle_bailian_error(self, e: Exception) -> None:
        """
        Check if the error is a specific Bailian API error and raise a more helpful message.
        """
        error_str = str(e)
        if "No generation chunks were returned" in error_str or "null value for 'choices'" in error_str:
            logger.error(f"Bailian API Error detected: {error_str}")
            raise ValueError(
                "Aliyun Bailian API returned an invalid response. "
                "This usually means authentication failed or the model is incorrect. "
                "Please check your API Key, Model Name, and permissions in Bailian Console."
            ) from e
