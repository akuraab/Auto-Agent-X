from __future__ import annotations

from typing import Any, Dict, List, Optional


class FAQService:
    def __init__(self):
        self._faq_items: List[Dict[str, Any]] = [
            {
                "id": "faq_stream_events",
                "question": "流式接口有哪些事件",
                "answer": "当前流式事件包括 status、intent、rewrite、faq_hit、portal、route、plan、tool_status、retrieval、token、done、error。",
                "keywords": ["stream", "事件", "sse", "token", "done"],
            },
            {
                "id": "faq_backend_entry",
                "question": "后端聊天入口在哪里",
                "answer": "后端入口在 /api/v1/chat 与 /api/v1/chat/stream，API 路由定义位于 backend/api/chat.py。",
                "keywords": ["入口", "chat", "api", "stream", "backend"],
            },
            {
                "id": "faq_intent_route",
                "question": "如何进行意图识别和路由",
                "answer": "系统先做意图识别，再由路由器按意图选择专用子代理，并动态加载对应提示词和工具。",
                "keywords": ["意图", "路由", "intent", "route", "agent"],
            },
            {
                "id": "faq_context_missing",
                "question": "为什么回答会提示上下文不足",
                "answer": "当检索证据不足或结果为空时，系统会显式说明上下文不足并提示补充文件名、模块名或错误栈。",
                "keywords": ["上下文", "证据", "检索", "不足", "clarification"],
            },
        ]

    async def match(self, query: str) -> Optional[Dict[str, Any]]:
        text = (query or "").lower()
        if not text:
            return None
        best_item: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for item in self._faq_items:
            score = self._score(text, item["question"], item["keywords"])
            if score > best_score:
                best_score = score
                best_item = item
        if best_item and best_score >= 0.72:
            return {
                "id": best_item["id"],
                "question": best_item["question"],
                "answer": best_item["answer"],
                "confidence": round(best_score, 3),
            }
        return None

    def _score(self, text: str, question: str, keywords: List[str]) -> float:
        question_hit = 0.5 if question in text or text in question else 0.0
        hits = sum(1 for key in keywords if key.lower() in text)
        keyword_score = min(0.5, hits / max(1, len(keywords)))
        return round(question_hit + keyword_score, 4)
