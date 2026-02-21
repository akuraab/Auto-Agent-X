from typing import Dict, List, Any, Optional
from jinja2 import Template, Environment, BaseLoader
from pydantic import BaseModel

class PromptTemplate(BaseModel):
    """Prompt Template"""
    id: str
    name: str
    description: str
    template: str
    variables: List[str]                    # Template variable list
    system_prompt: Optional[str] = None     # System prompt
    few_shot_examples: Optional[List[Dict]] = None  # Few-shot examples

class PromptService:
    """Prompt Service"""

    def __init__(self, template_store):
        self.template_store = template_store
        self.jinja_env = Environment(loader=BaseLoader())

        # Default system prompt
        self.default_system_prompt = """You are a professional code assistant, skilled in:
1. Understanding code logic and architecture
2. Providing clear technical explanations
3. Giving executable code suggestions

Please answer the question based on the provided context information. If the context is insufficient to answer the question, please state clearly."""

    async def assemble(
        self,
        template_id: str,
        query: str,
        context: Dict[str, Any],
        chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, str]:
        """
        Assemble prompt

        Args:
            template_id: Template ID
            query: User question
            context: Context information (retrieval results etc.)
            chat_history: Chat history

        Returns:
            {"system": system_prompt, "user": user_prompt}
        """
        # 1. Get template
        template = await self.template_store.get(template_id)
        if not template:
            # Fallback to a default template if not found
             template = PromptTemplate(
                id="default",
                name="Default",
                description="Default template",
                template="{{query}}\n\nContext:\n{{context}}",
                variables=["query", "context"]
            )

        # 2. Build variables
        variables = {
            "query": query,
            "context": self._format_context(context),
            "chat_history": self._format_history(chat_history) if chat_history else "",
            **context  # Unpack other context variables
        }

        # 3. Render user prompt
        user_prompt = self._render_template(template.template, variables)

        # 4. Build system prompt
        system_prompt = template.system_prompt or self.default_system_prompt

        # 5. Add Few-shot examples
        if template.few_shot_examples:
            system_prompt += "\n\nExamples:\n" + \
                self._format_few_shot(template.few_shot_examples)

        return {
            "system": system_prompt,
            "user": user_prompt
        }

    def _format_context(self, context: Dict) -> str:
        """Format retrieval context"""
        if "retrieval_results" not in context:
            return ""

        formatted = []
        for i, result in enumerate(context["retrieval_results"], 1):
            formatted.append(f"""
[Document {i}]
Source: {result.metadata.get('source', 'Unknown')}
Relevance: {result.score:.2f}
Content:
{result.content}
---""")

        return "\n".join(formatted)

    def _format_history(self, history: List[Dict]) -> str:
        """Format chat history"""
        formatted = []
        for msg in history[-5:]:  # Keep only last 5 turns
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def _render_template(self, template_str: str, variables: Dict) -> str:
        """Render template using Jinja2"""
        template = self.jinja_env.from_string(template_str)
        return template.render(**variables)

    def _format_few_shot(self, examples: List[Dict]) -> str:
        formatted = []
        for ex in examples:
            formatted.append(f"Q: {ex['input']}\nA: {ex['output']}")
        return "\n\n".join(formatted)

    async def optimize_for_token_limit(
        self,
        prompt: Dict[str, str],
        max_tokens: int,
        tokenizer
    ) -> Dict[str, str]:
        """
        Optimize prompt based on Token limit

        Strategy:
        1. Prioritize keeping system prompt
        2. Compress context (remove low relevance docs)
        3. Truncate chat history
        4. Finally truncate current query
        """
        # Calculate current Token count
        total_tokens = sum(
            len(tokenizer.encode(p))
            for p in prompt.values()
        )

        if total_tokens <= max_tokens:
            return prompt

        # Need compression
        # excess = total_tokens - max_tokens

        # Strategy 1: Remove low relevance docs
        # ... Implementation logic

        return prompt
