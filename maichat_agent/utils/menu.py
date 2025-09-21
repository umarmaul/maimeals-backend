from typing import Dict
from functools import lru_cache
import os

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from sqlalchemy import create_engine


class RecommendationInput(BaseModel):
    required_calories: float = Field(
        ..., description="Required calories for the meal in one day"
    )
    preferred_menu: str = Field(..., description="Preferred menu for the meal")


collection_name = "menu"


@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    # Ensure required env vars exist
    db_url = os.getenv("DB_URL")
    if not db_url:
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        name = os.getenv("DB_NAME")
        user = os.getenv("DB_USER")
        pwd = os.getenv("DB_PASSWORD")
        if not all([host, port, name, user, pwd]):
            raise RuntimeError(
                "Database environment variables are not fully set. Provide DB_URL or DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD."
            )
        db_url = f"postgresql+psycopg://{user}:{pwd}@{host}:{port}/{name}"

    engine = create_engine(
        db_url,
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
        pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "1800")),
        pool_pre_ping=True,
    )

    # OPENAI_API_KEY must be set in env (loaded by load_dotenv in server.start)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=engine,
        use_jsonb=True,
    )


@tool("menu-recommendation", args_schema=RecommendationInput, return_direct=True)
def menu_recommendation(required_calories: float, preferred_menu: str) -> list[Dict]:
    """Give Recommendation Process for a meal based on required calories"""
    vector_store = get_vector_store()
    results = vector_store.similarity_search(
        query=preferred_menu,
        k=3,
        filter={
            "calories": {"$lt": required_calories / 3},
        },
    )
    return [result.metadata for result in results]
