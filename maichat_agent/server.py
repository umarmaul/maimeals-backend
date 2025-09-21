import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langserve import validation as ls_validation
from fastapi.openapi.utils import get_openapi

# from maichat_agent.chain import create_graph  # MOVE THIS IMPORT INSIDE start()
from maichat_agent.schema import ChatInputType

import os
import logging
from typing import Any
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager


def start() -> None:
    load_dotenv()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    logger = logging.getLogger("maichat_server")

    # Import after .env is loaded so menu.py sees env vars
    from maichat_agent.chain import create_graph

    app = FastAPI(
        title="Generative UI Backend for mAIMeals Food Recommendation",
        version="1.0",
        description="This API provides a chat interface for the mAIMeals food recommendation system.",
    )

    def _rebuild_langserve_models():
        for name in dir(ls_validation):
            obj = getattr(ls_validation, name)
            rebuild = getattr(obj, "model_rebuild", None)
            if callable(rebuild):
                try:
                    rebuild(force=True)
                except Exception:
                    pass

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        _rebuild_langserve_models()
        logger.info("App startup completed.")
        try:
            yield
        finally:
            logger.info("App shutdown completed.")

    app.router.lifespan_context = lifespan

    def custom_openapi():
        _rebuild_langserve_models()
        if app.openapi_schema:
            return app.openapi_schema
        app.openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Consistent JSON error handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "validation_error",
                    "detail": exc.errors(),
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": "http_error",
                    "detail": exc.detail,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_error",
                    "detail": "Internal server error",
                }
            },
        )

    # Simple health endpoint
    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok"}

    graph = create_graph()
    add_routes(
        app,
        graph,
        path="/chat",
        input_type=ChatInputType,
        output_type=dict,
        playground_type="default",
    )

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7800"))
    # Configure Uvicorn loggers without using deprecated 'uvicorn.error'
    level_name = os.getenv("UVICORN_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    for logger_name in ("uvicorn", "uvicorn.access"):
        logging.getLogger(logger_name).setLevel(level)
    logger.info("Starting server on %s:%d ...", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()
