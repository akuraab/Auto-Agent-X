from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "Auto-Agent-X"
    API_V1_STR: str = "/api/v1"
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL_NAME: str = "qwen3.5-flash"
    
    # Vector DB
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Database
    DATABASE_URL: str = "sqlite:///./sql_app.db"  # Default to SQLite for now


settings = Settings()
