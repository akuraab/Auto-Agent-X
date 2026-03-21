from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, List, Optional


class QueryRewriteService:
    def __init__(self):
        self._expansion_rules = {
            "报错": ["error", "exception", "traceback"],
            "错误": ["error", "bug", "failure"],
            "接口": ["api", "endpoint", "handler", "route"],
            "登录": ["auth", "token", "session", "permission"],
            "性能": ["latency", "throughput", "slow", "timeout"],
            "数据库": ["sql", "mysql", "postgres", "query"],
            "检索": ["retrieve", "search", "recall", "rerank"],
            "路由": ["router", "route", "dispatch"],
            "意图": ["intent", "classification", "router"],
            "上下文": ["context", "prompt", "citation"],
            "日志": ["log", "trace", "monitor"],
            "修复": ["fix", "patch", "resolve"],
        }

    def rewrite(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raw = (query or "").strip()
        keywords = self._extract_keywords(raw)
        expansions = self._semantic_expansion(raw, keywords)
        time_range = self._normalize_time(raw)
        source = (context or {}).get("source", "unknown")
        rewritten = self._assemble_rewritten_query(raw, keywords, expansions, time_range)
        return {
            "original_query": raw,
            "rewritten_query": rewritten,
            "keywords": keywords,
            "expansions": expansions,
            "time_range": time_range,
            "context_source": source,
        }

    def _extract_keywords(self, query: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", query.lower())
        ordered: List[str] = []
        for token in tokens:
            if len(token) < 2:
                continue
            if token not in ordered:
                ordered.append(token)
            if len(ordered) >= 10:
                break
        return ordered

    def _semantic_expansion(self, query: str, keywords: List[str]) -> List[str]:
        lowered = query.lower()
        expansions: List[str] = []
        for key, values in self._expansion_rules.items():
            if key in query or key in lowered or key in keywords:
                for value in values:
                    if value not in expansions:
                        expansions.append(value)
        return expansions[:10]

    def _normalize_time(self, query: str) -> Dict[str, str]:
        now = datetime.now(timezone.utc)
        lowered = query.lower()
        if any(x in lowered for x in ["today", "今天"]):
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = now
            return {"start": start.isoformat(), "end": end.isoformat(), "label": "today"}
        if any(x in lowered for x in ["yesterday", "昨天"]):
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1) - timedelta(seconds=1)
            return {"start": start.isoformat(), "end": end.isoformat(), "label": "yesterday"}
        if any(x in lowered for x in ["本周", "this week", "最近", "recently"]):
            start = now - timedelta(days=7)
            return {"start": start.isoformat(), "end": now.isoformat(), "label": "last_7_days"}
        if any(x in lowered for x in ["本月", "this month"]):
            start = now - timedelta(days=30)
            return {"start": start.isoformat(), "end": now.isoformat(), "label": "last_30_days"}
        return {"start": "", "end": "", "label": "unspecified"}

    def _assemble_rewritten_query(
        self,
        raw: str,
        keywords: List[str],
        expansions: List[str],
        time_range: Dict[str, str],
    ) -> str:
        blocks = [raw]
        if keywords:
            blocks.append("keywords: " + ", ".join(keywords))
        if expansions:
            blocks.append("semantic expansion: " + ", ".join(expansions))
        if time_range.get("label") and time_range.get("label") != "unspecified":
            blocks.append(f"time window: {time_range.get('label')}")
        return " | ".join(blocks)
