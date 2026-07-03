"""
api/app.py
==========
The FastAPI application. This wires all the route modules together
and adds health-check, CORS, and exception-handling middleware.

HOW TO START THE API (for beginners):
  From the project root folder, run:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

  Then open http://localhost:8000/docs in your browser to see the full
  interactive API documentation where you can test every endpoint.

  The --reload flag makes the server restart automatically when you
  edit any Python file, which is useful during development.
"""

import os
import time
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.data     import router as data_router
from api.routes.training import router as training_router
from api.routes.inference import router as inference_router
from api.schemas import HealthResponse

# ---- Create the FastAPI app --------------------------------

app = FastAPI(
    title="RRIN — Restorative Retinal Imaging Network API",
    description="""
## Retinal Fundus Image Restoration API

This API exposes the full RRIN training and inference pipeline as REST endpoints.

### Quick Start
1. **Prepare data** → POST /api/v1/data/ingest (one call per dataset)
2. **Score images** → POST /api/v1/data/quality-score
3. **Create splits** → POST /api/v1/data/create-splits
4. **Train model**  → POST /api/v1/training/start
5. **Monitor**      → GET  /api/v1/training/status (poll this)
6. **Restore image** → POST /api/v1/inference/restore (upload a JPEG)

### Interactive Docs
Visit **/docs** for the full Swagger UI where you can test every endpoint.
Visit **/redoc** for the ReDoc alternative documentation.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---- CORS (allows browser clients to call the API) --------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # In production, replace with your specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Request timing middleware ----------------------------

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add X-Process-Time header to every response (for debugging)."""
    start_time = time.time()
    response   = await call_next(request)
    process_ms = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_ms:.1f}"
    return response


# ---- Global exception handler ----------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return a clean JSON error instead of an HTML stack trace."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# ---- Health check endpoint --------------------------------

@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Quick sanity check — confirms the API is running and shows
    whether the key dependencies (GPU, checkpoint, database) are available.
    """
    from src.config import CHECKPOINT_DIR, METADATA_DB_PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        device = f"cuda ({torch.cuda.get_device_name(0)})"

    return HealthResponse(
        status="ok",
        device=device,
        checkpoint_exists=os.path.exists(os.path.join(CHECKPOINT_DIR, "best.pt")),
        database_exists=os.path.exists(METADATA_DB_PATH),
        version="1.0.0",
    )


# ---- Register route modules --------------------------------

app.include_router(data_router,      prefix="/api/v1")
app.include_router(training_router,  prefix="/api/v1")
app.include_router(inference_router, prefix="/api/v1")


# ---- Run directly (python api/app.py) ----------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
