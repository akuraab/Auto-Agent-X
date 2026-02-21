# Auto-Agent-X: RAG-based AI Q&A System

This project is a RAG (Retrieval-Augmented Generation) based AI Q&A system.

## Project Structure

```
Auto-Agent-X/
├── app/                  # Backend Application (FastAPI)
│   ├── api/              # API Routes
│   ├── core/             # Core Configuration
│   ├── models/           # Data Models (Pydantic & SQLAlchemy)
│   ├── services/         # Business Logic Services
│   ├── engine/           # RAG Engine Components
│   ├── infrastructure/   # Infrastructure Clients (DB, Cache, etc.)
│   └── utils/            # Utility Functions
├── frontend/             # Frontend Application (React + Vite)
├── plan                  # Project Plan & Architecture
└── requirements.txt      # Python Dependencies
```

## Getting Started

### Backend Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```
