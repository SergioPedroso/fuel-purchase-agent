"""Configuração central, lida de variáveis de ambiente (prefixo FUEL_) e do .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODELS = {"gemini": "gemini-3.8-flash", "claude": "claude-opus-5-5"}
# Modelo reserva quando o principal está sobrecarregado (a Claude faz isso no servidor).
DEFAULT_FALLBACK_MODELS = {"gemini": "gemini-3.5-flash-lite", "claude": None}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FUEL_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    llm_provider: Literal["gemini", "claude"] = "gemini"
    gemini_api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    llm_model: str | None = None
    llm_fallback_model: str | None = None
    llm_effort: str = "medium"
    llm_max_tokens: int = 16000
    max_agent_steps: int = 8

    # Horizonte da decisão de compra (dias) e janela de notícias consideradas.
    horizon_days: int = 7
    news_max_age_days: int = 10
    news_per_query: int = 8

    # Histórico de mercado usado nos indicadores e no gráfico.
    market_history_days: int = 120

    logs_dir: Path = PROJECT_ROOT / "logs"
    outputs_dir: Path = PROJECT_ROOT / "outputs"

    @property
    def model_name(self) -> str:
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]

    @property
    def fallback_model_name(self) -> str | None:
        return self.llm_fallback_model or DEFAULT_FALLBACK_MODELS[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
