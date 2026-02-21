from fastapi import FastAPI
from backend.core.config import settings
from backend.core.logging import setup_logging
from backend.api import api_router

setup_logging()

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/health")
def health_check():
    return {"status": "ok"}
