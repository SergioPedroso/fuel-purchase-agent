"""Fixtures: mercado sintético e notícias com datas conhecidas."""

from datetime import UTC, date, datetime, timedelta

import pytest

from fuel_agent.config import Settings
from fuel_agent.market import build_snapshot
from fuel_agent.news import NewsArticle

TODAY = date(2026, 10, 8)
NOW = datetime(2026, 10, 8, 23, 59, tzinfo=UTC)


def _series(start_value: float, daily_change: float, days: int = 60) -> dict[date, float]:
    """Série diária com variação composta constante, terminando em TODAY."""
    first = TODAY - timedelta(days=days)
    return {
        first + timedelta(days=i): start_value * (1 + daily_change) ** i for i in range(days + 1)
    }


def make_market(oil_daily: float = 0.0, dollar_daily: float = 0.0):
    dollar = _series(5.0, dollar_daily)
    usd = {
        "brent_usd": _series(80.0, oil_daily),
        "gasolina_golfo_usd": _series(2.5, oil_daily),
        "diesel_golfo_usd": _series(3.0, oil_daily),
    }
    return build_snapshot(dollar, usd)


@pytest.fixture
def flat_market():
    return make_market()


@pytest.fixture
def rising_market():
    return make_market(oil_daily=0.01)  # ~+7% em 7 dias


def article(title: str, days_ago: float = 0, summary: str = "") -> NewsArticle:
    return NewsArticle(
        title=title,
        source="Fonte",
        published_at=NOW - timedelta(days=days_ago),
        url="https://example.org",
        summary=summary,
        origin="google:teste",
    )


@pytest.fixture
def settings(tmp_path):
    return Settings(
        logs_dir=tmp_path / "logs",
        outputs_dir=tmp_path / "outputs",
        gemini_api_key="test-key",
        anthropic_api_key="test-key",
    )
