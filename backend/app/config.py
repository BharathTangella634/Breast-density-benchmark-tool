from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    ground_truth_csv: Path = Path("data/private/ground_truth.csv")
    history_db: Path = Path("data/private/evaluation_history.db")
    image_manifest_csv: Path = Path("data/benchmark_prep/benchmark_test_public.csv")
    image_root: Path = Path("data/private/benchmark_test_png")
    onnx_input_size: int = 1024
    onnx_input_channels: int = 1
    max_csv_upload_mb: int = 25
    max_onnx_upload_mb: int = 750
    onnx_upload_dir: Path = Path("data/private/onnx_uploads")
    onnx_timeout_seconds: int = 3600
    allowed_labels: tuple[str, ...] = ("A", "B", "C", "D")
    enable_quadratic_kappa: bool = True

    database_url: str | None = None
    allowed_origins: list[str] = []
    gcs_bucket: str | None = None
    gcs_data_prefix: str = "benchmark"

    model_config = SettingsConfigDict(
        env_prefix="BENCHMARK_",
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
