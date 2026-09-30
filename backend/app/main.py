"""FastAPI application: REST API under /api + the web application at /.

Run:  python -m backend.app.main [--workspace DIR] [--port 8000]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def create_app():
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    from src.utils.logging import setup_logging

    from .api.routes import router
    from .core.settings import settings
    from .database.db import init_db
    from .services.datasets import discover
    from .services.explore import sync_cli_runs

    setup_logging()
    init_db()
    discover()
    sync_cli_runs()
    app = FastAPI(title="EntiQ — Business Identity Intelligence", version="2.0.0",
                  description="Entity resolution across three business-record sources.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"])

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; script-src 'self'; connect-src 'self'")
        return resp

    app.include_router(router)
    fe = settings.frontend_dir
    if fe.exists():
        app.mount("/static", StaticFiles(directory=fe), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(fe / "index.html")

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    if a.workspace:
        os.environ["ENTIQ_WORKSPACE"] = str(Path(a.workspace).resolve())
    import uvicorn

    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
else:  # `uvicorn backend.app.main:app`
    app = create_app()
