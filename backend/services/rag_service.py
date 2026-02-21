from typing import Dict, Any, AsyncIterator, List
import json
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable

from backend.core.logging import ThoughtProcessLogger
from backend.infrastructure.llm_client import LLMClient
from backend.services.intent_service import IntentService, IntentResult, IntentType
from backend.services.retrieval_service import RetrievalService

class RAGService:
    """
    RAG Service - Fully refactored to use LangChain Framework
    """
    def __init__(self):
        # Initialize dependencies
        self.llm_client = LLMClient()
        self.intent_service = IntentService(self.llm_client)
        self.retrieval_service = RetrievalService()
        self.llm = self.llm_client.get_chat_model()
        
        # Define Prompt Templates mapping
        self.prompt_templates = {
            "default_v1": ChatPromptTemplate.from_messages([
                ("system", """You are a professional code assistant.
Answer the question based ONLY on the following context.
If the answer is not in the context, say you don't know.

Context:
{context}"""),
                ("user", "{question}")
            ]),
            "code_search_v1": ChatPromptTemplate.from_messages([
                ("system", """You are a code search expert.
Based on the context provided, locate and explain the code relevant to the user's query.
Highlight file paths and line numbers if available.

Context:
{context}"""),
                ("user", "{question}")
            ]),
            "code_explain_v1": ChatPromptTemplate.from_messages([
                ("system", """You are a code explanation expert.
Explain the functionality of the code provided in the context.
Break down complex logic into simple steps.

Context:
{context}"""),
                ("user", "{question}")
            ]),
            "general_qa_v1": ChatPromptTemplate.from_messages([
                ("system", """You are a helpful technical assistant.
Answer the question using the provided context as a reference.

Context:
{context}"""),
                ("user", "{question}")
            ]),
            "casual_chat_v1": ChatPromptTemplate.from_messages([
                ("system", """You are a helpful technical assistant.
Answer the user's question directly and concisely."""),
                ("user", "{question}")
            ])
        }

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join([
            f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}"
            for d in docs
        ])

    async def chat(self, query: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """Process a chat request using LangChain LCEL"""
        
        # 0. Log Start
        ThoughtProcessLogger.log(session_id, "START_REQUEST", {"query": query})
        
        # 1. Intent Classification
        intent_result: IntentResult = await self.intent_service.classify(query)
        ThoughtProcessLogger.log(session_id, "INTENT_CLASSIFIED", intent_result.dict())
        
        # 2. Retrieval
        docs = []
        if intent_result.intent != IntentType.CASUAL_CHAT:
            try:
                docs = await self.retrieval_service.retrieve(query)
                ThoughtProcessLogger.log(session_id, "RETRIEVAL_SUCCESS", {
                    "count": len(docs),
                    "sources": [d.metadata.get("source") for d in docs]
                })
            except Exception as e:
                print(f"Retrieval failed: {e}")
                ThoughtProcessLogger.log(session_id, "RETRIEVAL_FAILED", {"error": str(e)})
                # Proceed with empty docs
        else:
            ThoughtProcessLogger.log(session_id, "RETRIEVAL_SKIPPED", {"reason": "casual_chat intent"})
        
        # 3. Select Prompt
        template_id = intent_result.suggested_prompt_template
        prompt_template = self.prompt_templates.get(template_id, self.prompt_templates["default_v1"])
        ThoughtProcessLogger.log(session_id, "PROMPT_SELECTED", {"template_id": template_id})
        
        # 4. Build Chain
        chain = prompt_template | self.llm | StrOutputParser()
        
        # 5. Execute Chain
        context_str = self._format_docs(docs)
        ThoughtProcessLogger.log(session_id, "LLM_INPUT", {
            "template_id": template_id,
            "context_preview": context_str[:500] + "..." if len(context_str) > 500 else context_str,
            "question": query
        })
        
        response_text = await chain.ainvoke({
            "context": context_str,
            "question": query
        })
        
        ThoughtProcessLogger.log(session_id, "LLM_OUTPUT", {"response": response_text})
        
        return {
            "session_id": session_id,
            "response": response_text,
            "intent": intent_result.dict(),
            "citations": [
                {
                    "source": d.metadata.get("source"),
                    "content": d.page_content[:100] + "...",
                    "relevance": d.metadata.get("score", 0.0)
                }
                for d in docs
            ],
            "metadata": {"template_used": template_id}
        }

    async def chat_stream(self, query: str, session_id: str, **kwargs) -> AsyncIterator[str]:
        """Process a chat request (streaming) using LangChain LCEL"""
        
        # 0. Log Start
        ThoughtProcessLogger.log(session_id, "START_REQUEST_STREAM", {"query": query})
        
        # 1. Intent Classification (Yield status)
        yield self._format_sse("status", {"stage": "intent_classify"})
        intent_result: IntentResult = await self.intent_service.classify(query)
        ThoughtProcessLogger.log(session_id, "INTENT_CLASSIFIED", intent_result.dict())
        
        if intent_result.requires_clarification:
            yield self._format_sse("clarification", {
                "message": "Clarification needed",
                "suggestions": [] 
            })
            # Even if clarification is needed, we should probably still try to answer or ask for it in a normal way
            # return  <-- Removed this return to allow the flow to continue to retrieval and generation


        # 2. Retrieval (Yield status)
        yield self._format_sse("status", {"stage": "retrieving"})
        
        docs = []
        if intent_result.intent != IntentType.CASUAL_CHAT:
            try:
                docs = await self.retrieval_service.retrieve(query)
                ThoughtProcessLogger.log(session_id, "RETRIEVAL_SUCCESS", {
                    "count": len(docs),
                    "sources": [d.metadata.get("source") for d in docs]
                })
            except Exception as e:
                print(f"Retrieval failed: {e}")
                ThoughtProcessLogger.log(session_id, "RETRIEVAL_FAILED", {"error": str(e)})
                yield self._format_sse("error", {"message": f"Retrieval failed: {str(e)}"})
                # Proceed with empty docs
        else:
            ThoughtProcessLogger.log(session_id, "RETRIEVAL_SKIPPED", {"reason": "casual_chat intent"})
        
        yield self._format_sse("retrieval", {
            "found": len(docs),
            "sources": [d.metadata.get("source") for d in docs]
        })

        # 3. Select Prompt
        template_id = intent_result.suggested_prompt_template
        prompt_template = self.prompt_templates.get(template_id, self.prompt_templates["default_v1"])
        ThoughtProcessLogger.log(session_id, "PROMPT_SELECTED", {"template_id": template_id})
        
        # 4. Build Chain
        chain = prompt_template | self.llm | StrOutputParser()
        
        # 5. Stream Generation
        yield self._format_sse("status", {"stage": "generating"})
        
        context_str = self._format_docs(docs)
        ThoughtProcessLogger.log(session_id, "LLM_INPUT", {
            "template_id": template_id,
            "context_preview": context_str[:500] + "..." if len(context_str) > 500 else context_str,
            "question": query
        })

        full_response = ""
        async for chunk in chain.astream({
            "context": context_str,
            "question": query
        }):
            full_response += chunk
            yield self._format_sse("token", {"content": chunk})
        
        ThoughtProcessLogger.log(session_id, "LLM_OUTPUT_STREAM_COMPLETE", {"response_length": len(full_response)})

        # 6. Completion
        yield self._format_sse("done", {
            "session_id": session_id,
            "citations": [
                {
                    "source": d.metadata.get("source"),
                    "content": d.page_content[:100] + "...",
                    "relevance": d.metadata.get("score", 0.0)
                }
                for d in docs
            ]
        })

    def _format_sse(self, event: str, data: Dict) -> str:
        """Format SSE message"""
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"
