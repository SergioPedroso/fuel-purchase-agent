"""Notícias que podem mover o preço do combustível: Google News (por tema) e MinasPetro.

Notícias são conteúdo externo e não confiável: aqui elas são reduzidas a campos simples
(título, fonte, data, link, resumo em texto puro e truncado). A triagem contra injeção de
instruções e dados pessoais acontece em `guardrails`, antes de o texto chegar ao LLM.
"""

import html
import logging
import re
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from urllib.parse import quote_plus

import feedparser
from pydantic import BaseModel

from fuel_agent.http import http_client

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
MINASPETRO_FEED = "https://minaspetro.com.br/feed/"
MAX_SUMMARY_CHARS = 400
TAG_RE = re.compile(r"<[^>]+>")
SPACES_RE = re.compile(r"\s+")


class NewsTopic(StrEnum):
    PETROBRAS = "petrobras"
    IMPOSTOS = "impostos"
    GEOPOLITICA = "geopolitica"
    ETANOL = "etanol"
    DISTRIBUICAO = "distribuicao"


# Consultas fixas: o LLM escolhe o tema, nunca o texto da busca.
TOPIC_QUERIES = {
    NewsTopic.PETROBRAS: (
        "Petrobras (reajuste OR aumento OR redução) (gasolina OR diesel) distribuidoras"
    ),
    NewsTopic.IMPOSTOS: '(ICMS OR "PIS/Cofins" OR CIDE OR desoneração OR tributação) combustíveis',
    NewsTopic.GEOPOLITICA: '(petróleo OR Brent) (guerra OR OPEP OR "Oriente Médio" OR sanções)',
    NewsTopic.ETANOL: "etanol (usinas OR safra OR Paulínia OR anidro OR hidratado) preço",
    NewsTopic.DISTRIBUICAO: (
        "(distribuidoras OR Betim OR Paulínia OR REGAP OR REPLAN) combustível preço"
    ),
}


class NewsArticle(BaseModel):
    title: str
    source: str
    published_at: datetime | None
    url: str
    summary: str
    origin: str  # "google:<tema>" ou "minaspetro"


def _clean_text(raw: str, limit: int) -> str:
    text = SPACES_RE.sub(" ", html.unescape(TAG_RE.sub(" ", raw or ""))).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _published(entry) -> datetime | None:
    parsed = entry.get("published_parsed")
    return datetime(*parsed[:6], tzinfo=UTC) if parsed else None


def parse_feed(
    content: bytes | str,
    *,
    origin: str,
    default_source: str = "",
    max_age_days: int,
    limit: int,
    now: datetime | None = None,
) -> list[NewsArticle]:
    feed = feedparser.parse(content)
    min_date = (now or datetime.now(UTC)) - timedelta(days=max_age_days)
    seen: set[str] = set()
    articles: list[NewsArticle] = []
    for entry in feed.entries:
        published = _published(entry)
        if published and published < min_date:
            continue
        source = (entry.get("source", {}) or {}).get("title", "") or default_source
        title = _clean_text(entry.get("title", ""), 200)
        # O Google News acrescenta " - Fonte" ao título; remove para deduplicar.
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        key = title.casefold()
        if not title or key in seen:
            continue
        seen.add(key)
        summary = _clean_text(entry.get("summary", ""), MAX_SUMMARY_CHARS)
        if summary.casefold().startswith(key):
            summary = ""  # o resumo do Google News repete título + fonte
        articles.append(
            NewsArticle(
                title=title,
                source=_clean_text(source, 80),
                published_at=published,
                url=entry.get("link", ""),
                summary=summary,
                origin=origin,
            )
        )
    articles.sort(key=lambda a: a.published_at or datetime.min.replace(tzinfo=UTC), reverse=True)
    return articles[:limit]


def _get(url: str, timeout: float) -> bytes:
    with http_client(timeout) as client:
        response = client.get(url)
    response.raise_for_status()
    return response.content


def search_topic(
    topic: NewsTopic, *, max_age_days: int, limit: int, timeout: float = 20
) -> list[NewsArticle]:
    query = quote_plus(f"{TOPIC_QUERIES[topic]} when:{max_age_days}d")
    content = _get(GOOGLE_NEWS_RSS.format(query=query), timeout)
    return parse_feed(content, origin=f"google:{topic}", max_age_days=max_age_days, limit=limit)


def fetch_minaspetro(*, max_age_days: int, limit: int, timeout: float = 20) -> list[NewsArticle]:
    content = _get(MINASPETRO_FEED, timeout)
    return parse_feed(
        content,
        origin="minaspetro",
        default_source="MinasPetro",
        max_age_days=max_age_days,
        limit=limit,
    )
