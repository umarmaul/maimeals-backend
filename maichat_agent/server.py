import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langserve import validation as ls_validation
from fastapi.openapi.utils import get_openapi

from maichat_agent.chain import create_graph
from maichat_agent.schema import ChatInputType


def start() -> None:
    load_dotenv()

    app = FastAPI(
        title="Generative UI Backend for mAIMeals Food Recommendation",
        version="1.0",
        description="This API provides a chat interface for the mAIMeals food recommendation system.",
    )

    # Rebuild all Pydantic models in langserve.validation
    def _rebuild_langserve_models():
        for name in dir(ls_validation):
            obj = getattr(ls_validation, name)
            rebuild = getattr(obj, "model_rebuild", None)
            if callable(rebuild):
                try:
                    rebuild(force=True)
                except Exception:
                    pass

    _rebuild_langserve_models()

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

    # Configure CORS
    origins = [
        "http://localhost",
        "http://localhost:7900",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    graph = create_graph()

    runnable = graph

    add_routes(
        app,
        runnable,
        path="/chat",
        input_type=ChatInputType,
        output_type=dict,
        playground_type="default",
    )
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=7800)


if __name__ == "__main__":
    start()
