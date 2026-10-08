"""Cálculo determinístico do sinal de compra por produto.

O LLM classifica as notícias (categoria, direção, intensidade, produtos afetados); o código
abaixo transforma essas classificações e os indicadores de mercado em uma pontuação de -1
(o preço tende a cair) a +1 (o preço tende a subir) e em uma decisão:

    pontuação >= +LIMIAR  -> ANTECIPAR a compra (preço deve subir)
    pontuação <= -LIMIAR  -> ADIAR a compra (preço deve cair)
    entre os dois         -> MANTER ROTINA

Antecipar ou adiar tem custo (capital, estoque, risco de errar). Por isso, quando mercado e
notícias apontam forte para lados opostos, a decisão é MANTER ROTINA com confiança baixa,
mesmo que a soma passe do limiar.

Eventos confirmados de reajuste da Petrobras ou de tributos, com vigência dentro do horizonte
(ex.: reajuste anunciado a partir de amanhã; imposto que volta em data definida), prevalecem
sobre o resto, porque já não são tendência, e sim fato.
"""

import math
from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from fuel_agent.market import MarketSnapshot
from fuel_agent.news import NewsArticle
from fuel_agent.products import PRODUCTS, ProductKey

DECISION_THRESHOLD = 0.2
STRONG_SIGNAL = 0.5
# Mercado e notícias opostos, cada um com pelo menos este peso: sinais conflitantes.
CONFLICT_MIN = 0.1
# Variação que conta como sinal "cheio" (pontuação 1) em 7 e em 30 dias.
FULL_MOVE_7D = 0.03
FULL_MOVE_30D = 0.08
NEWS_HALF_LIFE_DAYS = 4.0
CATEGORY_SATURATION = 1.0  # soma de pesos que leva um tema a ~±0,76 (tanh)


class Category(StrEnum):
    PETROBRAS = "petrobras_reajuste"
    IMPOSTOS = "impostos_tributos"
    GEOPOLITICA = "geopolitica_oferta"
    ETANOL = "etanol_safra"
    DISTRIBUICAO = "distribuicao_abastecimento"
    OUTROS = "outros"


class Direction(StrEnum):
    ALTA = "alta"
    BAIXA = "baixa"
    NEUTRA = "neutra"


class Intensity(StrEnum):
    FORTE = "forte"
    MODERADA = "moderada"
    FRACA = "fraca"


class Decision(StrEnum):
    ANTECIPAR = "ANTECIPAR"
    ADIAR = "ADIAR"
    MANTER = "MANTER ROTINA"


class Confidence(StrEnum):
    ALTA = "alta"
    MEDIA = "média"
    BAIXA = "baixa"


INTENSITY_WEIGHT = {Intensity.FORTE: 1.0, Intensity.MODERADA: 0.5, Intensity.FRACA: 0.25}

# Peso de cada tema na pontuação de notícias. Geopolítica pesa menos porque o efeito no
# petróleo já entra pelos indicadores de mercado (evita contar duas vezes): mesmo saturada,
# ela sozinha não cruza o limite de decisão; precisa de confirmação do mercado.
CATEGORY_WEIGHT = {
    Category.PETROBRAS: 1.0,
    Category.IMPOSTOS: 1.0,
    Category.DISTRIBUICAO: 0.7,
    Category.ETANOL: 0.7,
    Category.GEOPOLITICA: 0.3,
    Category.OUTROS: 0.2,
}
# Só fatos que mudam diretamente o preço de compra podem definir a decisão sozinhos.
DECISIVE_CATEGORIES = {Category.PETROBRAS, Category.IMPOSTOS}
DIRECTION_SIGN = {Direction.ALTA: 1, Direction.BAIXA: -1, Direction.NEUTRA: 0}


class NewsAssessment(BaseModel):
    """Classificação de uma notícia feita pelo LLM (formato obrigatório)."""

    id: int
    relevante: bool = Field(description="Afeta o preço de compra na distribuidora em 7 dias?")
    categoria: Category
    direcao: Direction = Field(description="Efeito provável sobre o preço na distribuidora")
    intensidade: Intensity
    produtos: list[ProductKey] = Field(description="Produtos afetados")
    evento_confirmado: bool = Field(
        description="Fato já decidido com data de vigência (ex.: reajuste anunciado), não opinião"
    )
    vigencia_no_horizonte: bool = Field(description="O efeito começa dentro do horizonte")
    motivo: str = Field(description="Uma frase explicando a classificação")


class NewsClassification(BaseModel):
    avaliacoes: list[NewsAssessment]


class Driver(BaseModel):
    source: str  # "mercado" ou "noticia"
    description: str
    contribution: float  # efeito na pontuação final


class ProductSignal(BaseModel):
    product: ProductKey
    decision: Decision
    confidence: Confidence
    score: float
    market_score: float
    news_score: float
    confirmed_event: str | None = None
    conflicting_signals: bool = False
    drivers: list[Driver]


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def market_component(product: ProductKey, market: MarketSnapshot) -> tuple[float, list[Driver]]:
    """Média ponderada das variações (7d pesa 70%, 30d pesa 30%) dos indicadores do produto."""
    spec = PRODUCTS[product]
    total, drivers = 0.0, []
    for key, weight in spec.market_weights.items():
        indicator = market.indicators[key]
        move_7 = _clamp((indicator.change_7d or 0) / FULL_MOVE_7D)
        move_30 = _clamp((indicator.change_30d or 0) / FULL_MOVE_30D)
        score = weight * (0.7 * move_7 + 0.3 * move_30)
        total += score
        drivers.append(
            Driver(
                source="mercado",
                description=f"{indicator.name}: {_pct(indicator.change_7d)} em 7 dias, "
                f"{_pct(indicator.change_30d)} em 30 dias",
                contribution=score * spec.market_share,
            )
        )
    return _clamp(total), drivers


def _recency_weight(published: datetime | None, now: datetime) -> float:
    if published is None:
        return 0.5
    age_days = max(0.0, (now - published).total_seconds() / 86400)
    return 0.5 ** (age_days / NEWS_HALF_LIFE_DAYS)


def news_component(
    product: ProductKey,
    assessments: list[NewsAssessment],
    articles: list[NewsArticle],
    now: datetime,
) -> tuple[float, list[Driver], list[NewsAssessment]]:
    """Pontuação de notícias por tema, para não contar várias vezes o mesmo fato.

    Em cada categoria, soma direção x intensidade x recência e satura com tanh (dez notícias
    sobre o mesmo assunto valem um tema forte, não dez sinais). As categorias são então
    combinadas com pesos (`CATEGORY_WEIGHT`). Também devolve os eventos confirmados que podem
    definir a decisão (só reajuste da Petrobras e tributos, com vigência no horizonte).
    """
    spec = PRODUCTS[product]
    sums: dict[Category, float] = defaultdict(float)
    items: list[tuple[NewsAssessment, float]] = []
    confirmed = []
    for item in assessments:
        if not item.relevante or product not in item.produtos or item.direcao == Direction.NEUTRA:
            continue
        weight = (
            DIRECTION_SIGN[item.direcao]
            * INTENSITY_WEIGHT[item.intensidade]
            * _recency_weight(articles[item.id - 1].published_at, now)
        )
        if item.evento_confirmado and item.vigencia_no_horizonte:
            weight *= 2
            if item.categoria in DECISIVE_CATEGORIES:
                confirmed.append(item)
        sums[item.categoria] += weight
        items.append((item, weight))

    category_scores = {c: math.tanh(s / CATEGORY_SATURATION) for c, s in sums.items()}
    total = sum(CATEGORY_WEIGHT[c] * score for c, score in category_scores.items())

    drivers = []
    for item, weight in items:
        # Parcela do item na pontuação do seu tema, proporcional ao seu peso bruto.
        share = weight / sums[item.categoria] if sums[item.categoria] else 0.0
        contribution = CATEGORY_WEIGHT[item.categoria] * category_scores[item.categoria] * share
        drivers.append(
            Driver(
                source="noticia",
                description=f"[{item.id}] {articles[item.id - 1].title}",
                contribution=contribution * (1 - spec.market_share),
            )
        )
    return _clamp(total), drivers, confirmed


def confirmed_direction(events: list[NewsAssessment]) -> int:
    """Saldo dos eventos confirmados: +1 alta, -1 baixa, 0 se empatarem ou não houver."""
    net = sum(DIRECTION_SIGN[e.direcao] * INTENSITY_WEIGHT[e.intensidade] for e in events)
    return 0 if abs(net) < 1e-9 else (1 if net > 0 else -1)


def _decide(score: float) -> Decision:
    if score >= DECISION_THRESHOLD:
        return Decision.ANTECIPAR
    if score <= -DECISION_THRESHOLD:
        return Decision.ADIAR
    return Decision.MANTER


def signals_conflict(market_part: float, news_part: float) -> bool:
    """Mercado e notícias puxam forte para lados opostos (partes já ponderadas)."""
    return market_part * news_part < 0 and min(abs(market_part), abs(news_part)) >= CONFLICT_MIN


def _confidence(score: float, market_part: float, news_part: float) -> Confidence:
    agree = market_part * news_part > 0
    if abs(score) >= STRONG_SIGNAL and agree:
        return Confidence.ALTA
    if abs(score) >= DECISION_THRESHOLD:
        return Confidence.MEDIA
    return Confidence.BAIXA


def compute_signal(
    product: ProductKey,
    market: MarketSnapshot,
    assessments: list[NewsAssessment],
    articles: list[NewsArticle],
    now: datetime | None = None,
) -> ProductSignal:
    now = now or datetime.now(UTC)
    spec = PRODUCTS[product]
    market_score, market_drivers = market_component(product, market)
    news_score, news_drivers, confirmed = news_component(product, assessments, articles, now)
    market_part = spec.market_share * market_score
    news_part = (1 - spec.market_share) * news_score
    score = market_part + news_part
    conflict = False

    direction = confirmed_direction(confirmed)
    if direction:
        # Fatos com data dentro do horizonte definem a direção, com confiança alta.
        score = direction * max(abs(score), STRONG_SIGNAL)
        decision, confidence = _decide(score), Confidence.ALTA
    elif confirmed:
        # Eventos confirmados em direções opostas se anulam: não há base para agir.
        decision, confidence = Decision.MANTER, Confidence.BAIXA
    elif signals_conflict(market_part, news_part):
        # Antecipar ou adiar tem custo; com evidências contraditórias, mantém a rotina.
        decision, confidence, conflict = Decision.MANTER, Confidence.BAIXA, True
    else:
        decision = _decide(score)
        confidence = _confidence(score, market_part, news_part)
    event = "; ".join(f"[{e.id}] {articles[e.id - 1].title}" for e in confirmed) or None

    drivers = sorted(market_drivers + news_drivers, key=lambda d: abs(d.contribution), reverse=True)
    return ProductSignal(
        product=product,
        decision=decision,
        confidence=confidence,
        score=round(score, 3),
        market_score=round(market_score, 3),
        news_score=round(news_score, 3),
        confirmed_event=event,
        conflicting_signals=conflict,
        drivers=drivers[:6],
    )


def _pct(value: float | None) -> str:
    return "n/d" if value is None else f"{value * 100:+.1f}%".replace(".", ",")
