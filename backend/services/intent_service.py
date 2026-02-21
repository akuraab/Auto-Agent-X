from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSerializable
from backend.infrastructure.llm_client import LLMClient

class IntentType(str, Enum):
    """Intent type enumeration"""
    CODE_SEARCH = "code_search"           # Code Search
    CODE_EXPLAIN = "code_explain"         # Code Explanation
    CODE_REVIEW = "code_review"           # Code Review
    BUG_FIX = "bug_fix"                   # Bug Fix
    REFACTOR = "refactor"                 # Refactoring Suggestion
    GENERAL_QA = "general_qa"             # General Q&A
    CASUAL_CHAT = "casual_chat"           # Casual Chat / Simple Greeting
    CLARIFICATION = "clarification"       # Clarification Needed

class IntentResult(BaseModel):
    """Intent recognition result"""
    intent: IntentType = Field(description="The classified intent type")
    confidence: float = Field(description="Confidence score between 0 and 1")
    entities: Dict[str, Any] = Field(description="Extracted entities relevant to the intent")
    suggested_prompt_template: str = Field(description="Suggested prompt template ID")
    requires_clarification: bool = Field(description="Whether clarification is needed from the user")

class IntentService:
    """Intent Recognition Service (LangChain Implementation)"""

    def __init__(self, llm_client: LLMClient, prompt_templates: Dict = None):
        self.llm = llm_client.get_chat_model(temperature=0.1)
        self.templates = prompt_templates or {}
        self.allowed_template_ids = {
            "default_v1",
            "code_search_v1",
            "code_explain_v1",
            "code_review_v1",
            "bug_fix_v1",
            "refactor_v1",
            "general_qa_v1",
            "casual_chat_v1"
        }
        
        # Define the parser
        self.parser = JsonOutputParser(pydantic_object=IntentResult)
        
        # Define the prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the user question to identify the intent type and key entities.

Available Intent Types:
- code_search: Search for specific code
- code_explain: Explain code functionality
- code_review: Review code quality
- bug_fix: Fix code issues
- refactor: Refactoring suggestions
- general_qa: General technical Q&A
- casual_chat: Simple greetings, small talk, or basic questions like "1+1" that don't need context

Return the result in JSON format matching the following structure:
{{
    "intent": "intent_type",
    "confidence": 0.95,
    "entities": {{
        "language": "programming_language",
        "component": "component/module",
        "keywords": ["keyword1", "keyword2"]
    }},
    "suggested_prompt_template": "template_id",
    "requires_clarification": false
}}

If the confidence is low (< 0.7), set requires_clarification to true.
"""),
            ("user", "{query}")
        ])

        # Create the chain
        self.chain = self.prompt | self.llm | self.parser

    async def classify(self, query: str, context: Optional[Dict] = None) -> IntentResult:
        """
        Main classification method using LangChain
        """
        try:
            result = await self.chain.ainvoke({"query": query})
            
            # Post-process to ensure type safety and defaults
            intent_str = result.get("intent", "general_qa")
            # Map string to Enum safely
            intent_enum = IntentType.GENERAL_QA
            if intent_str in IntentType._value2member_map_:
                intent_enum = IntentType(intent_str)
            
            # Determine template if not provided by LLM or validate it
            template_id = result.get("suggested_prompt_template")
            if (not template_id 
                or template_id == "template_id" 
                or template_id not in self.allowed_template_ids):
                template_id = self._match_template(intent_str)

            return IntentResult(
                intent=intent_enum,
                confidence=result.get("confidence", 0.0),
                entities=result.get("entities", {}),
                suggested_prompt_template=template_id,
                requires_clarification=result.get("requires_clarification", False)
            )
            
        except Exception as e:
            # Fallback for parsing errors or LLM failures
            print(f"Intent classification error: {e}")
            return IntentResult(
                intent=IntentType.GENERAL_QA,
                confidence=0.5,
                entities={},
                suggested_prompt_template="general_qa_v1",
                requires_clarification=False
            )

    def _match_template(self, intent: str) -> str:
        """Match prompt template based on intent"""
        template_map = {
            "code_search": "code_search_v1",
            "code_explain": "code_explain_v1",
            "code_review": "code_review_v1",
            "bug_fix": "bug_fix_v1",
            "refactor": "refactor_v1",
            "general_qa": "general_qa_v1",
            "casual_chat": "casual_chat_v1"
        }
        return template_map.get(intent, "default_v1")
