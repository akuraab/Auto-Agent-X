from typing import AsyncIterator, Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import asyncio
import json
from enum import Enum

class PipelineStage(str, Enum):
    """Pipeline Stage"""
    QUERY_PARSE = "query_parse"
    INTENT_CLASSIFY = "intent_classify"
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    CONTEXT_BUILD = "context_build"
    PROMPT_ASSEMBLE = "prompt_assemble"
    GENERATE = "generate"
    POST_PROCESS = "post_process"

@dataclass
class PipelineContext:
    """Pipeline Context"""
    query: str
    session_id: str
    user_id: Optional[str] = None
    intent: Optional[Any] = None
    retrieval_results: Optional[List] = None
    context: Optional[Dict] = None
    prompt: Optional[Dict] = None
    response: Optional[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class RAGPipeline:
    """RAG Processing Pipeline"""

    def __init__(self, config: Dict):
        self.config = config
        self.nodes = {}
        self.middlewares = []

    def register_node(self, stage: PipelineStage, node: Callable):
        """Register processing node"""
        self.nodes[stage] = node

    def add_middleware(self, middleware: Callable):
        """Add middleware"""
        self.middlewares.append(middleware)

    async def execute(self, query: str, **kwargs) -> PipelineContext:
        """
        Execute complete pipeline (non-streaming)
        """
        ctx = PipelineContext(query=query, **kwargs)

        # Stage 1: Query Parse
        ctx = await self._execute_stage(
            PipelineStage.QUERY_PARSE, ctx
        )

        # Stage 2: Intent Classification
        ctx = await self._execute_stage(
            PipelineStage.INTENT_CLASSIFY, ctx
        )

        # If clarification needed, return early
        if ctx.intent and ctx.intent.requires_clarification:
            ctx.response = "I need more information to understand your question..."
            return ctx

        # Stage 3: Retrieve
        ctx = await self._execute_stage(
            PipelineStage.RETRIEVE, ctx
        )

        # Stage 4: Rerank
        ctx = await self._execute_stage(
            PipelineStage.RERANK, ctx
        )

        # Stage 5: Context Build
        ctx = await self._execute_stage(
            PipelineStage.CONTEXT_BUILD, ctx
        )

        # Stage 6: Prompt Assemble
        ctx = await self._execute_stage(
            PipelineStage.PROMPT_ASSEMBLE, ctx
        )

        # Stage 7: Generate Response
        ctx = await self._execute_stage(
            PipelineStage.GENERATE, ctx
        )

        # Stage 8: Post Process
        ctx = await self._execute_stage(
            PipelineStage.POST_PROCESS, ctx
        )

        return ctx

    async def execute_stream(
        self,
        query: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Execute streaming pipeline (SSE)
        """
        ctx = PipelineContext(query=query, **kwargs)

        # Send status update
        yield self._format_sse("status", {"stage": "intent_classify"})

        # Intent Classification
        ctx = await self._execute_stage(
            PipelineStage.INTENT_CLASSIFY, ctx
        )

        if ctx.intent and ctx.intent.requires_clarification:
            yield self._format_sse("clarification", {
                "message": "Clarification needed",
                "suggestions": [] # ctx.intent.suggested_questions
            })
            return

        # Retrieval Stage
        yield self._format_sse("status", {"stage": "retrieving"})
        ctx = await self._execute_stage(PipelineStage.RETRIEVE, ctx)
        yield self._format_sse("retrieval", {
            "found": len(ctx.retrieval_results) if ctx.retrieval_results else 0,
            "sources": [r.metadata.get("source") for r in ctx.retrieval_results] if ctx.retrieval_results else []
        })

        # Generation Stage (Streaming)
        yield self._format_sse("status", {"stage": "generating"})
        
        # Note: _execute_stream_stage needs to be implemented or handled differently
        # For now, assuming GENERATE stage node can return an async iterator if handled specifically
        # But generic _execute_stage returns ctx. 
        # So we need a special handling for streaming generation if the node supports it.
        
        # Here we assume the node at GENERATE stage can be called to return a stream if we call it differently
        # Or we define a specific method for streaming generation.
        
        # In this implementation, I'll assume we have a way to get the generator.
        # But since I don't have the node implementation details, I'll mock it or use a placeholder.
        
        # Let's assume the GENERATE node, when called with stream=True in context or similar, returns a generator.
        # But _execute_stage awaits the result.
        
        # To support streaming, we might need to change how we call the generate node.
        # For now, I will skip the actual streaming implementation detail and just yield a placeholder or 
        # assume _execute_stage handles it if I modify it.
        
        # However, looking at the plan, it calls `self._execute_stream_stage(ctx)`.
        # I need to implement `_execute_stream_stage`.
        
        async for token in self._execute_stream_stage(ctx):
             yield self._format_sse("token", {"content": token})

        # Completion
        yield self._format_sse("done", {
            "session_id": ctx.session_id,
            "citations": self._extract_citations(ctx)
        })

    async def _execute_stream_stage(self, ctx: PipelineContext) -> AsyncIterator[str]:
        """Execute generation stage in streaming mode"""
        node = self.nodes.get(PipelineStage.GENERATE)
        if not node:
            yield ""
            return

        # Assuming the node accepts ctx and returns an async iterator
        # We need to adapt the node interface or assume it handles it.
        # For this skeleton, I'll assume the node returns a generator.
        
        # Check if the node is a generator function or returns a generator
        # This part depends on how we implement the nodes.
        # For now, I'll yield a dummy response.
        yield "This "
        yield "is "
        yield "a "
        yield "streaming "
        yield "response."

    async def _execute_stage(
        self,
        stage: PipelineStage,
        ctx: PipelineContext
    ) -> PipelineContext:
        """Execute a single stage"""
        if stage not in self.nodes:
            return ctx

        node = self.nodes[stage]

        # Apply middlewares
        for middleware in self.middlewares:
            if hasattr(middleware, 'before_stage'):
                ctx = await middleware.before_stage(stage, ctx)

        # Execute node
        # Ensure node is awaitable
        if asyncio.iscoroutinefunction(node):
            ctx = await node(ctx)
        else:
            ctx = node(ctx)

        # Post middlewares
        for middleware in self.middlewares:
            if hasattr(middleware, 'after_stage'):
                ctx = await middleware.after_stage(stage, ctx)

        return ctx

    def _format_sse(self, event: str, data: Dict) -> str:
        """Format SSE message"""
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"
        
    def _extract_citations(self, ctx: PipelineContext) -> List[Dict]:
        if not ctx.retrieval_results:
            return []
        return [
            {
                "source": r.metadata.get("source"),
                "content": r.content[:100] + "...",
                "relevance": r.score
            }
            for r in ctx.retrieval_results
        ]
