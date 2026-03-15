"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    cohere_api_key: str
    cohere_model: str = "command-r7b-12-2024"
    port: int = 8080

    model_config = {"env_file": ".env"}


def get_settings() -> Settings:
    return Settings()
