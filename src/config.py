"""Configuration settings for HeartBeat AI."""
from pathlib import Path

# pydantic v2.12 moved BaseSettings to the separate `pydantic-settings` package.
# Try the standard import first, and fall back to the compatibility package when needed.
try:
    from pydantic import BaseSettings
except Exception:
    from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_name: str = "HeartBeat AI"
    sample_rate: int = 4000
    segment_seconds: float = 5.0
    n_mels: int = 128
    model_dir: Path = Path("models")
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")
    metadata_file: Path = Path("models/heartbeat_metadata.json")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
