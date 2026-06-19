from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    ground_truth_csv: Path = Path("data/private/ground_truth.csv")
    history_db: Path = Path("data/private/evaluation_history.db")
    allowed_labels: tuple[str, ...] = ("A", "B", "C", "D")
    enable_quadratic_kappa: bool = True

    model_config = SettingsConfigDict(
        env_prefix="BENCHMARK_",
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
