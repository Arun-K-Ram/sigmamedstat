"""
CRIP-X FastAPI Application

Entry point for the API layer.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import reliability, fixtures
from api.schemas.response import HealthResponse
from crip_x.utils.config import settings
from crip_x.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="CRIP-X API",
    description=(
        "Contextual Reliability Intelligence Platform — "
        "real-time medical device signal trustworthiness evaluation"
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────
# Allow React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Alternative React port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(reliability.router)
app.include_router(fixtures.router)


# ── Health Check ──────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="0.1.0",
        pipeline_ready=True,
    )


@app.get("/")
async def root():
    return {
        "name": "CRIP-X API",
        "version": "0.1.0",
        "docs": "/docs",
    }