"""
Luma API Demo Server

Runs the full API layer (chat, insights, insight-moments, teacher, personalization)
with mock core dependencies — no database required.

Start with:
    python demo.py

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

import uvicorn
from unittest.mock import MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from luma.api.controllers.chat_controller import router as chat_router, set_luma_service, get_luma_service
from luma.api.controllers.insight_controller import router as insight_router
from luma.api.controllers.teacher_controller import router as teacher_router
from luma.api.controllers.personalization_controller import router as personalization_router
from luma.api.middleware.logging import LoggingMiddleware
from luma.api.middleware.error_handler import ErrorHandlerMiddleware
from luma.api.services.luma_service import LumaService
from luma.api.factory import build_llm_engine
from luma.core.structured_logger import StructuredLogger


# ---------------------------------------------------------------------------
# Build a mock LumaService with realistic demo responses
# ---------------------------------------------------------------------------

def build_demo_service() -> LumaService:
    memory = MagicMock()
    memory.retrieve.return_value = {
        "memories": [
            {
                "id": "mem_001",
                "content": "User prefers concise technical explanations.",
                "metadata": {"category": "preference"},
                "timestamp": "2026-04-15T10:00:00",
                "category": "preference",
                "tags": ["style"],
            }
        ],
        "total_count": 1,
        "query_metadata": {},
    }
    memory.store.return_value = "mem_new"

    adaptation_ctx = MagicMock()
    adaptation_ctx.tone = "technical"
    adaptation_ctx.style = "concise"
    adaptation_ctx.focus = "deep-technical"
    adaptation_ctx.reasons = {
        "tone": "User has a technical background based on past interactions.",
        "style": "User prefers short, direct answers.",
        "focus": "User consistently asks deep implementation questions.",
    }
    adaptation_ctx.model_dump.return_value = {
        "tone": "technical",
        "style": "concise",
        "focus": "deep-technical",
        "reasons": adaptation_ctx.reasons,
    }

    personalization_result = MagicMock()
    personalization_result.adaptation = adaptation_ctx

    personalization_engine = MagicMock()
    personalization_engine.personalize.return_value = personalization_result

    insight_report = MagicMock()
    insight_report.insights = [
        {"text": "You frequently ask about Python async patterns.", "confidence": 0.91},
        {"text": "You tend to explore topics in depth before moving on.", "confidence": 0.85},
    ]

    insight_engine = MagicMock()
    insight_engine.generate_insights.return_value = insight_report

    insight_moments_engine = MagicMock()
    insight_moments_engine.generate_moments.return_value = [
        {"message": "You've been studying this topic for 3 sessions — great consistency!"},
    ]

    teaching_session = MagicMock()
    teaching_session.session_id = "sess-demo-001"
    teaching_session.status = "active"
    teaching_session.lessons = [
        {"id": "l1", "title": "Introduction to the topic", "difficulty": "beginner"},
        {"id": "l2", "title": "Core concepts", "difficulty": "intermediate"},
    ]
    teaching_session.explanations = [
        {"lesson_id": "l1", "content": "Let's start with the fundamentals..."},
        {"lesson_id": "l2", "content": "Building on that, the core idea is..."},
    ]
    teaching_session.exercises = [
        {"id": "e1", "lesson_id": "l1", "prompt": "Explain the concept in your own words."},
    ]

    teacher_mode = MagicMock()
    teacher_mode.teach.return_value = teaching_session

    # Build LLMEngine using the factory (will be None if OPENAI_API_KEY not set)
    logger = StructuredLogger()
    llm_engine = build_llm_engine(logger)

    return LumaService(
        memory_interface=memory,
        insight_engine=insight_engine,
        insight_moments_engine=insight_moments_engine,
        personalization_engine=personalization_engine,
        teacher_mode=teacher_mode,
        llm_engine=llm_engine,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Build the demo app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Luma AI — Demo",
    description=(
        "Live demo of the Luma API layer.\n\n"
        "All endpoints use mock data — no database required.\n\n"
        "**Endpoints:**\n"
        "- `POST /api/v1/chat` — Send a message, get a contextual response\n"
        "- `GET  /api/v1/insights` — Retrieve generated insights\n"
        "- `GET  /api/v1/insight-moments` — Retrieve triggered insight moments\n"
        "- `POST /api/v1/teacher/start` — Start a teaching session\n"
        "- `POST /api/v1/teacher/continue` — Continue a teaching session\n"
        "- `GET  /api/v1/personalization` — Get user personalization profile\n"
    ),
    version="0.1.0-demo",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(LoggingMiddleware)

from fastapi import APIRouter
api_router = APIRouter()
api_router.include_router(chat_router)
api_router.include_router(insight_router)
api_router.include_router(teacher_router)
api_router.include_router(personalization_router)
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Luma demo is running", "docs": "http://127.0.0.1:8000/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy", "mode": "demo"}


# Inject the mock service
set_luma_service(build_demo_service())


if __name__ == "__main__":
    print("\n🚀  Luma demo starting...")
    print("   Swagger UI  →  http://127.0.0.1:8000/docs")
    print("   Health      →  http://127.0.0.1:8000/health\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
