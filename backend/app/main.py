from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import subprocess

from fastapi import FastAPI

from .search.bm25 import BM25Index
from .search.vector import VectorIndex
from .routes import router


logger = logging.getLogger(__name__)

APP_VERSION = os.getenv("APP_VERSION", "0.1.0")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()
        return commit if commit else "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


APP_COMMIT = _get_git_commit()


def _try_load_index(name: str, index_obj):
    try:
        index_obj.load()
        logger.info("Loaded %s index from %s", name, index_obj.index_dir)
        return index_obj
    except FileNotFoundError as exc:
        logger.warning("%s index files missing; continuing without it: %s", name, exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bm25_index = _try_load_index("BM25", BM25Index())
    app.state.vector_index = _try_load_index("Vector", VectorIndex())
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "commit": APP_COMMIT,
    }
