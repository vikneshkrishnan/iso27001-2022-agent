from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Document Extraction System"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")

    # File Upload Configuration
    max_file_size: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    allowed_extensions: List[str] = Field(
        default=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
        env="ALLOWED_EXTENSIONS"
    )
    upload_dir: str = Field(default="uploads", env="UPLOAD_DIR")
    output_dir: str = Field(default="data", env="OUTPUT_DIR")

    # Database Configuration
    database_url: str = Field(default="sqlite:///./documents.db", env="DATABASE_URL")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")

    # Celery Configuration
    celery_broker_url: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")
    celery_max_retries: int = Field(default=3, env="CELERY_MAX_RETRIES")
    celery_retry_delay: int = Field(default=60, env="CELERY_RETRY_DELAY")

    # Processing Configuration
    max_parallel_tasks: int = Field(default=10, env="MAX_PARALLEL_TASKS")
    task_timeout: int = Field(default=300, env="TASK_TIMEOUT")  # 5 minutes

    # OCR Configuration
    tesseract_path: Optional[str] = Field(default=None, env="TESSERACT_PATH")
    ocr_languages: List[str] = Field(default=["eng"], env="OCR_LANGUAGES")
    ocr_confidence_threshold: float = Field(default=0.6, env="OCR_CONFIDENCE_THRESHOLD")

    # PDF Processing Configuration
    pdf_dpi: int = Field(default=300, env="PDF_DPI")
    pdf_max_pages: int = Field(default=100, env="PDF_MAX_PAGES")

    # NLP Configuration
    spacy_model: str = Field(default="en_core_web_sm", env="SPACY_MODEL")
    ner_confidence_threshold: float = Field(default=0.8, env="NER_CONFIDENCE_THRESHOLD")

    # Monitoring Configuration
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # CrewAI Configuration
    crewai_max_execution_time: int = Field(default=120, env="CREWAI_MAX_EXECUTION_TIME")
    crewai_verbose: bool = Field(default=False, env="CREWAI_VERBOSE")

    # Quality Validation Configuration
    min_extraction_confidence: float = Field(default=0.7, env="MIN_EXTRACTION_CONFIDENCE")
    min_text_length: int = Field(default=10, env="MIN_TEXT_LENGTH")

    # LLM API Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")

    # Pinecone Configuration
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="us-east-1", env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="iso27001", env="PINECONE_INDEX_NAME")
    pinecone_dimension: int = Field(default=1024, env="PINECONE_DIMENSION")  # Match Pinecone index

    # Embedding Configuration
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(default=100, env="EMBEDDING_BATCH_SIZE")
    max_embedding_tokens: int = Field(default=8191, env="MAX_EMBEDDING_TOKENS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent.parent.parent

    @property
    def upload_path(self) -> Path:
        return self.base_dir / self.upload_dir

    @property
    def output_path(self) -> Path:
        return self.base_dir / self.output_dir

    @property
    def logs_path(self) -> Path:
        return self.base_dir / "logs"

    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.upload_path.mkdir(exist_ok=True)
        self.output_path.mkdir(exist_ok=True)
        self.logs_path.mkdir(exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()