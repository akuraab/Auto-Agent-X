from __future__ import annotations
import asyncio
import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.core.logging import ThoughtProcessLogger
from backend.infrastructure.llm_client import LLMClient
from backend.services.faq_service import FAQService
from backend.services.intent_service import IntentResult, IntentService, IntentType
from backend.services.query_rewrite_service import QueryRewriteService
from backend.services.retrieval_service import RetrievalService


ToolCallable = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]


@dataclass
class AgentProfile:
    name: str
    prompt_id: str
    tools: List[str]


@dataclass
class RouteDecision:
    route: str
    profile: AgentProfile


@dataclass
class RuntimeState:
    session_id: str
    query: str
    rewritten_query: str
    rewrite_meta: Dict[str, Any]
    intent: IntentResult
    route: str
    prompt_id: str
    loaded_tools: List[str]
    context: Dict[str, Any]
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    faq_hit: Optional[Dict[str, Any]] = None
    tool_outputs: Dict[str, Any] = field(default_factory=dict)
    portal_trace: List[str] = field(default_factory=list)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolCallable] = {}

    def register(self, name: str, tool: ToolCallable) -> None:
        self._tools[name] = tool

    def resolve(self, tool_names: List[str]) -> Dict[str, ToolCallable]:
        return {name: self._tools[name] for name in tool_names if name in self._tools}


class AgentRouter:
    def __init__(self):
        self._profiles = {
            "portal_clarifier": AgentProfile(name="portal_clarifier", prompt_id="clarifier_v1", tools=[]),
            "code_navigator": AgentProfile(name="code_navigator", prompt_id="code_search_v2", tools=["retrieve_code_context", "open_top_source"]),
            "code_explainer": AgentProfile(name="code_explainer", prompt_id="code_explain_v2", tools=["retrieve_code_context", "open_top_source"]),
            "bug_solver": AgentProfile(name="bug_solver", prompt_id="bug_fix_v2", tools=["retrieve_code_context", "open_top_source"]),
            "code_reviewer": AgentProfile(name="code_reviewer", prompt_id="code_review_v2", tools=["retrieve_code_context", "open_top_source"]),
            "generalist": AgentProfile(name="generalist", prompt_id="general_qa_v2", tools=["retrieve_code_context"]),
            "chat_companion": AgentProfile(name="chat_companion", prompt_id="casual_chat_v2", tools=[]),
        }
        self._template_route_map = {
            "code_search_v1": "code_navigator",
            "code_explain_v1": "code_explainer",
            "bug_fix_v1": "bug_solver",
            "code_review_v1": "code_reviewer",
            "refactor_v1": "code_reviewer",
            "general_qa_v1": "generalist",
            "casual_chat_v1": "chat_companion",
            "default_v1": "generalist",
        }

    def route(self, intent: IntentResult) -> RouteDecision:
        if intent.intent == IntentType.CLARIFICATION:
            return RouteDecision(route="portal_clarifier", profile=self._profiles["portal_clarifier"])
        if intent.requires_clarification and intent.confidence < 0.7:
            return RouteDecision(route="portal_clarifier", profile=self._profiles["portal_clarifier"])
        route_from_template = self._template_route_map.get(intent.suggested_prompt_template)
        if route_from_template and route_from_template in self._profiles:
            return RouteDecision(route=route_from_template, profile=self._profiles[route_from_template])
        if intent.intent == IntentType.CODE_SEARCH:
            return RouteDecision(route="code_navigator", profile=self._profiles["code_navigator"])
        if intent.intent == IntentType.CODE_EXPLAIN:
            return RouteDecision(route="code_explainer", profile=self._profiles["code_explainer"])
        if intent.intent == IntentType.BUG_FIX:
            return RouteDecision(route="bug_solver", profile=self._profiles["bug_solver"])
        if intent.intent in {IntentType.CODE_REVIEW, IntentType.REFACTOR}:
            return RouteDecision(route="code_reviewer", profile=self._profiles["code_reviewer"])
        if intent.intent == IntentType.CASUAL_CHAT:
            return RouteDecision(route="chat_companion", profile=self._profiles["chat_companion"])
        return RouteDecision(route="generalist", profile=self._profiles["generalist"])


class PromptResourceLoader:
    def __init__(self):
        self._prompts = {
            "clarifier_v1": """你是门户Agent。当前问题信息不足，请先提出关键澄清问题，并给出继续分析所需最少输入。""",
            "code_search_v2": """你是代码定位专家。请先给出结论，再给出定位依据和后续建议。仅依据上下文回答。""",
            "code_explain_v2": """你是代码讲解专家。先说明整体职责，再拆关键流程与边界条件。仅依据上下文回答。""",
            "bug_fix_v2": """你是排障专家。先给根因假设，再给验证步骤和修复方案。仅依据上下文回答。""",
            "code_review_v2": """你是代码审查专家。先输出风险点，再给可执行改进建议。仅依据上下文回答。""",
            "general_qa_v2": """你是技术助手。请基于上下文准确回答，若上下文不足明确指出。""",
            "casual_chat_v2": """你是专业且简洁的技术助手，直接回答用户问题。""",
        }

    def load(self, prompt_id: str) -> str:
        return self._prompts.get(prompt_id, self._prompts["general_qa_v2"])


class SpecialistAgent:
    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client.get_chat_model()

    @staticmethod
    def _build_context(state: RuntimeState) -> str:
        docs: List[Document] = state.tool_outputs.get("retrieve_code_context", {}).get("docs", [])
        snippets = []
        for doc in docs[:8]:
            snippets.append(
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "score": float(doc.metadata.get("score", 0.0)),
                    "type": doc.metadata.get("type", ""),
                    "content": doc.page_content,
                }
            )
        opened = state.tool_outputs.get("open_top_source", {}).get("file_content", "")
        payload = {
            "query": state.query,
            "rewritten_query": state.rewritten_query,
            "time_range": state.rewrite_meta.get("time_range", {}),
            "retrieval_docs": snippets,
            "top_source_content": opened,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _build_citations(state: RuntimeState) -> List[Dict[str, Any]]:
        docs: List[Document] = state.tool_outputs.get("retrieve_code_context", {}).get("docs", [])
        return [
            {
                "source": d.metadata.get("source"),
                "content": d.page_content[:120] + ("..." if len(d.page_content) > 120 else ""),
                "relevance": float(d.metadata.get("score", 0.0)),
            }
            for d in docs[:8]
        ]

    async def run(self, state: RuntimeState, prompt_text: str) -> Dict[str, Any]:
        if state.route == "portal_clarifier":
            response = "我需要更多信息来准确处理：请提供目标文件/模块名、期望动作（定位/解释/修复）以及你当前遇到的现象。"
            return {
                "response": response,
                "citations": [],
            }
        context = self._build_context(state)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_text + "\n\nContext:\n{context}"),
                ("user", "{question}"),
            ]
        )
        chain = prompt | self._llm | StrOutputParser()
        try:
            response = await chain.ainvoke({"context": context, "question": state.query})
        except Exception:
            if context:
                response = f"已完成多智能体路由与检索，但生成阶段失败。当前最相关上下文如下：\n\n{context[:1200]}"
            else:
                response = "已完成多智能体路由，但未检索到足够上下文，请补充更具体的代码线索。"
        return {
            "response": response,
            "citations": self._build_citations(state),
        }

    async def run_stream(self, state: RuntimeState, prompt_text: str) -> AsyncIterator[str]:
        if state.route == "portal_clarifier":
            yield "我需要更多信息来准确处理：请提供目标文件/模块名、期望动作（定位/解释/修复）以及你当前遇到的现象。"
            return
        context = self._build_context(state)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_text + "\n\nContext:\n{context}"),
                ("user", "{question}"),
            ]
        )
        chain = prompt | self._llm | StrOutputParser()
        try:
            async for chunk in chain.astream({"context": context, "question": state.query}):
                if not chunk:
                    continue
                yield chunk
        except Exception:
            if context:
                yield f"已完成多智能体路由与检索，但生成阶段失败。当前最相关上下文如下：\n\n{context[:1200]}"
            else:
                yield "已完成多智能体路由，但未检索到足够上下文，请补充更具体的代码线索。"


class PortalAgent:
    def build_trace(self, state: RuntimeState) -> List[str]:
        intent_text = state.intent.intent.value if isinstance(state.intent.intent, IntentType) else str(state.intent.intent)
        return [
            f"门户Agent接管请求: {state.query[:80]}",
            f"识别意图: {intent_text} (置信度 {state.intent.confidence:.2f})",
            f"查询改写: {state.rewritten_query[:120]}",
            f"自动路由到: {state.route}",
            f"动态加载提示词: {state.prompt_id}",
            f"动态加载工具: {', '.join(state.loaded_tools) if state.loaded_tools else '无'}",
        ]

    async def stream_trace(self, state: RuntimeState) -> AsyncIterator[Dict[str, Any]]:
        for item in self.build_trace(state):
            yield {"phase": "portal_progress", "message": item}


class MultiAgentOrchestrator:
    def __init__(self):
        self.llm_client = LLMClient()
        self.intent_service = IntentService(self.llm_client)
        self.query_rewrite_service = QueryRewriteService()
        self.faq_service = FAQService()
        self.retrieval_service = RetrievalService()
        self.router = AgentRouter()
        self.prompt_loader = PromptResourceLoader()
        self.portal_agent = PortalAgent()
        self.specialist_agent = SpecialistAgent(self.llm_client)
        self.tool_registry = ToolRegistry()
        self._register_tools()

    def _register_tools(self) -> None:
        async def retrieve_code_context(query: str, runtime: Dict[str, Any]) -> Dict[str, Any]:
            docs = await self.retrieval_service.retrieve(query)
            return {"docs": docs}

        async def open_top_source(query: str, runtime: Dict[str, Any]) -> Dict[str, Any]:
            docs: List[Document] = runtime.get("docs", [])
            if not docs:
                return {"file_content": ""}
            top_source = docs[0].metadata.get("source")
            if not top_source:
                return {"file_content": ""}
            path = Path(top_source)
            if not path.exists():
                return {"file_content": ""}
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                return {"file_content": ""}
            return {"file_content": content[:1500]}

        self.tool_registry.register("retrieve_code_context", retrieve_code_context)
        self.tool_registry.register("open_top_source", open_top_source)

    def _build_plan(self, route: str, loaded_tools: List[str]) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = [{"name": "route_decision", "type": "portal"}]
        for tool_name in loaded_tools:
            steps.append({"name": tool_name, "type": "tool"})
        steps.append({"name": "specialist_generate", "type": "llm"})
        if route == "portal_clarifier":
            return [{"name": "clarification", "type": "portal"}]
        return steps

    def _validate_result(self, state: RuntimeState, response: str, citations: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = (response or "").strip()
        flags: List[str] = []
        if not text:
            text = "当前未生成有效回复，请补充目标文件名、模块名或报错堆栈。"
            flags.append("empty_response_fallback")
        if state.route in {"code_navigator", "code_explainer", "bug_solver", "code_reviewer", "generalist"} and not citations:
            text = "未检索到可验证证据，以下是基于当前输入的保守建议：\n" + text
            flags.append("null_evidence_explained")
        if state.route == "bug_solver" and ("根因" not in text or "修复" not in text):
            text = f"根因分析：\n{text}\n\n修复建议：\n- 基于检索结果逐步验证调用链与错误触发条件。\n- 先添加最小复现与回归校验，再提交修复。"
            flags.append("bugfix_template_enforced")
        return {"response": text, "flags": flags}

    async def _prepare_state(self, query: str, session_id: str, context: Optional[Dict[str, Any]] = None) -> RuntimeState:
        ThoughtProcessLogger.log(session_id, "PORTAL_START", {"query": query})
        safe_context = context or {}
        rewrite_meta = self.query_rewrite_service.rewrite(query, safe_context)
        rewritten_query = rewrite_meta.get("rewritten_query") or query
        intent_task = self.intent_service.classify(rewritten_query, context=safe_context)
        faq_task = self.faq_service.match(rewritten_query)
        intent, faq_hit = await asyncio.gather(intent_task, faq_task)
        decision = self.router.route(intent)
        prompt_text = self.prompt_loader.load(decision.profile.prompt_id)
        loaded_tools = decision.profile.tools
        state = RuntimeState(
            session_id=session_id,
            query=query,
            rewritten_query=rewritten_query,
            rewrite_meta=rewrite_meta,
            intent=intent,
            route=decision.route,
            prompt_id=decision.profile.prompt_id,
            loaded_tools=loaded_tools,
            context=safe_context,
            faq_hit=faq_hit,
            plan_steps=self._build_plan(decision.route, loaded_tools),
        )
        state.portal_trace = self.portal_agent.build_trace(state)
        ThoughtProcessLogger.log(
            session_id,
            "PORTAL_ROUTE_READY",
            {
                "intent": intent.intent.value if isinstance(intent.intent, IntentType) else str(intent.intent),
                "route": state.route,
                "prompt_id": state.prompt_id,
                "tools": state.loaded_tools,
                "faq_hit": bool(state.faq_hit),
            },
        )
        state.tool_outputs["prompt_text"] = {"value": prompt_text}
        return state

    async def _execute_tools(self, state: RuntimeState) -> None:
        runtime = {"docs": []}
        resolved = self.tool_registry.resolve(state.loaded_tools)
        for name, tool in resolved.items():
            result = await tool(state.rewritten_query, runtime)
            state.tool_outputs[name] = result
            if name == "retrieve_code_context":
                runtime["docs"] = result.get("docs", [])

    async def run(self, query: str, session_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state = await self._prepare_state(query=query, session_id=session_id, context=context)
        if state.faq_hit:
            return {
                "session_id": session_id,
                "response": state.faq_hit["answer"],
                "intent": state.intent.dict(),
                "citations": [],
                "metadata": {
                    "route": "faq_shortcut",
                    "prompt_id": "faq_builtin_v1",
                    "loaded_tools": [],
                    "portal_trace": state.portal_trace,
                    "faq_hit": state.faq_hit,
                    "rewrite": state.rewrite_meta,
                    "plan_steps": [{"name": "faq_shortcut", "type": "portal"}],
                    "validation_flags": [],
                },
            }
        if state.route == "portal_clarifier":
            specialist = await self.specialist_agent.run(state=state, prompt_text=state.tool_outputs["prompt_text"]["value"])
            validated = self._validate_result(state, specialist["response"], specialist["citations"])
            return {
                "session_id": session_id,
                "response": validated["response"],
                "intent": state.intent.dict(),
                "citations": [],
                "metadata": {
                    "route": state.route,
                    "prompt_id": state.prompt_id,
                    "loaded_tools": state.loaded_tools,
                    "portal_trace": state.portal_trace,
                    "faq_hit": None,
                    "rewrite": state.rewrite_meta,
                    "plan_steps": state.plan_steps,
                    "validation_flags": validated["flags"],
                },
            }
        await self._execute_tools(state)
        prompt_text = state.tool_outputs["prompt_text"]["value"]
        specialist = await self.specialist_agent.run(state=state, prompt_text=prompt_text)
        validated = self._validate_result(state, specialist["response"], specialist["citations"])
        response = validated["response"]
        citations = specialist["citations"]
        return {
            "session_id": session_id,
            "response": response,
            "intent": state.intent.dict(),
            "citations": citations,
            "metadata": {
                "route": state.route,
                "prompt_id": state.prompt_id,
                "loaded_tools": state.loaded_tools,
                "portal_trace": state.portal_trace,
                "faq_hit": None,
                "rewrite": state.rewrite_meta,
                "plan_steps": state.plan_steps,
                "validation_flags": validated["flags"],
            },
        }

    async def run_stream(self, query: str, session_id: str, context: Optional[Dict[str, Any]] = None) -> AsyncIterator[str]:
        def fmt(event: str, data: Dict[str, Any]) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        state = await self._prepare_state(query=query, session_id=session_id, context=context)
        yield fmt("status", {"stage": "intent_classify"})
        yield fmt("rewrite", state.rewrite_meta)
        yield fmt("intent", state.intent.dict())
        yield fmt("plan", {"steps": state.plan_steps})
        if state.faq_hit:
            yield fmt("faq_hit", state.faq_hit)
            yield fmt(
                "done",
                {
                    "session_id": session_id,
                    "response": state.faq_hit["answer"],
                    "citations": [],
                    "metadata": {
                        "route": "faq_shortcut",
                        "prompt_id": "faq_builtin_v1",
                        "loaded_tools": [],
                        "portal_trace": state.portal_trace,
                        "faq_hit": state.faq_hit,
                        "rewrite": state.rewrite_meta,
                        "plan_steps": [{"name": "faq_shortcut", "type": "portal"}],
                        "validation_flags": [],
                    },
                },
            )
            return
        async for item in self.portal_agent.stream_trace(state):
            yield fmt("portal", item)
        if state.route == "portal_clarifier":
            clarifier = await self.specialist_agent.run(state=state, prompt_text=state.tool_outputs["prompt_text"]["value"])
            validated = self._validate_result(state, clarifier["response"], [])
            yield fmt("clarification", {"message": validated["response"], "suggestions": []})
            yield fmt(
                "done",
                {
                    "session_id": session_id,
                    "response": validated["response"],
                    "citations": [],
                    "metadata": {
                        "route": state.route,
                        "prompt_id": state.prompt_id,
                        "loaded_tools": state.loaded_tools,
                        "portal_trace": state.portal_trace,
                        "faq_hit": None,
                        "rewrite": state.rewrite_meta,
                        "plan_steps": state.plan_steps,
                        "validation_flags": validated["flags"],
                    },
                },
            )
            return
        yield fmt("status", {"stage": "retrieving"})
        yield fmt("route", {"agent": state.route})
        resolved = self.tool_registry.resolve(state.loaded_tools)
        runtime = {"docs": []}
        for name, tool in resolved.items():
            yield fmt("tool_status", {"tool": name, "status": "running"})
            result = await tool(state.rewritten_query, runtime)
            state.tool_outputs[name] = result
            if name == "retrieve_code_context":
                docs = result.get("docs", [])
                runtime["docs"] = docs
                yield fmt(
                    "retrieval",
                    {"found": len(docs), "sources": [d.metadata.get("source") for d in docs[:8]]},
                )
            yield fmt("tool_status", {"tool": name, "status": "success"})
        state.tool_outputs["prompt_text"] = {"value": self.prompt_loader.load(state.prompt_id)}
        yield fmt("status", {"stage": "generating"})
        full = ""
        async for chunk in self.specialist_agent.run_stream(state=state, prompt_text=state.tool_outputs["prompt_text"]["value"]):
            full += chunk
            yield fmt("token", {"content": chunk})
        citations = SpecialistAgent._build_citations(state)
        validated = self._validate_result(state, full, citations)
        yield fmt(
            "done",
            {
                "session_id": session_id,
                "response": validated["response"],
                "citations": citations,
                "metadata": {
                    "route": state.route,
                    "prompt_id": state.prompt_id,
                    "loaded_tools": state.loaded_tools,
                    "portal_trace": state.portal_trace,
                    "faq_hit": None,
                    "rewrite": state.rewrite_meta,
                    "plan_steps": state.plan_steps,
                    "validation_flags": validated["flags"],
                },
            },
        )
