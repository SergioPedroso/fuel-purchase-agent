import pytest
from conftest import NOW, article, make_market

from fuel_agent.products import ProductKey
from fuel_agent.signals import (
    Category,
    Confidence,
    Decision,
    Direction,
    Intensity,
    NewsAssessment,
    compute_signal,
)

GAS = ProductKey.GASOLINA_COMUM
DIESEL = ProductKey.DIESEL_S10


def assessment(
    id_: int,
    direction: Direction,
    *,
    category: Category = Category.PETROBRAS,
    intensity: Intensity = Intensity.FORTE,
    products=(GAS, DIESEL),
    confirmed: bool = False,
) -> NewsAssessment:
    return NewsAssessment(
        id=id_,
        relevante=True,
        categoria=category,
        direcao=direction,
        intensidade=intensity,
        produtos=list(products),
        evento_confirmado=confirmed,
        vigencia_no_horizonte=confirmed,
        motivo="teste",
    )


def test_flat_market_and_no_news_keeps_routine(flat_market):
    signal = compute_signal(GAS, flat_market, [], [], NOW)
    assert signal.decision == Decision.MANTER
    assert signal.score == 0


def test_rising_international_prices_push_to_anticipate():
    market = make_market(oil_daily=0.02)  # ~+15% em 7 dias
    signal = compute_signal(DIESEL, market, [], [], NOW)
    assert signal.market_score == pytest.approx(0.8)  # derivados no teto; dólar (20%) parado
    assert signal.decision == Decision.ANTECIPAR


def test_falling_prices_and_bearish_news_push_to_postpone():
    market = make_market(oil_daily=-0.01)
    articles = [article("Petrobras reduz preço do diesel")]
    news = [assessment(1, Direction.BAIXA, products=[DIESEL])]
    signal = compute_signal(DIESEL, market, news, articles, NOW)
    assert signal.decision == Decision.ADIAR


def test_repeated_news_on_same_theme_do_not_stack(flat_market):
    """Dez notícias sobre o mesmo tema valem um tema forte, não dez sinais."""
    articles = [article(f"Petróleo sobe com guerra {i}") for i in range(10)]
    news = [assessment(i + 1, Direction.ALTA, category=Category.GEOPOLITICA) for i in range(10)]
    signal = compute_signal(GAS, flat_market, news, articles, NOW)
    # Geopolítica pesa 0,3: saturada, sozinha não cruza o limite de decisão.
    assert signal.news_score <= 0.3
    assert signal.decision == Decision.MANTER


def test_conflicting_market_and_news_keep_routine():
    """Mercado em queda forte e notícias de alta forte: não há base para antecipar."""
    market = make_market(oil_daily=-0.005)
    articles = [article("Governo avalia elevar PIS/Cofins"), article("Petrobras sinaliza alta")]
    news = [
        assessment(1, Direction.ALTA, category=Category.IMPOSTOS),
        assessment(2, Direction.ALTA, category=Category.PETROBRAS),
    ]
    signal = compute_signal(GAS, market, news, articles, NOW)
    assert signal.market_score < -0.5
    assert signal.score > 0.2  # a soma, sozinha, indicaria ANTECIPAR
    assert signal.decision == Decision.MANTER
    assert signal.confidence == Confidence.BAIXA
    assert signal.conflicting_signals


def test_confirmed_petrobras_increase_overrides_market():
    market = make_market(oil_daily=-0.01)  # mercado sugere queda
    articles = [article("Petrobras reajusta gasolina a partir de amanhã")]
    news = [assessment(1, Direction.ALTA, confirmed=True)]
    signal = compute_signal(GAS, market, news, articles, NOW)
    assert signal.decision == Decision.ANTECIPAR
    assert signal.confidence == Confidence.ALTA
    assert "[1]" in signal.confirmed_event


def test_confirmed_geopolitical_event_does_not_override(flat_market):
    articles = [article("EUA impõem sanções ao Irã")]
    news = [assessment(1, Direction.ALTA, category=Category.GEOPOLITICA, confirmed=True)]
    signal = compute_signal(DIESEL, flat_market, news, articles, NOW)
    assert signal.confirmed_event is None
    assert signal.confidence != Confidence.ALTA


def test_opposite_confirmed_events_lower_confidence(flat_market):
    articles = [article("Imposto volta dia 10"), article("Petrobras reduz preço dia 10")]
    news = [
        assessment(1, Direction.ALTA, category=Category.IMPOSTOS, confirmed=True),
        assessment(2, Direction.BAIXA, confirmed=True),
    ]
    signal = compute_signal(GAS, flat_market, news, articles, NOW)
    assert signal.confidence == Confidence.BAIXA


def test_old_news_weighs_less(flat_market):
    news = [assessment(1, Direction.ALTA, category=Category.IMPOSTOS)]
    recent = compute_signal(GAS, flat_market, news, [article("ICMS sobe", days_ago=0)], NOW)
    old = compute_signal(GAS, flat_market, news, [article("ICMS sobe", days_ago=8)], NOW)
    assert recent.news_score > old.news_score > 0


def test_news_only_affects_listed_products(flat_market):
    articles = [article("Safra ruim encarece etanol")]
    news = [assessment(1, Direction.ALTA, category=Category.ETANOL, products=[ProductKey.ETANOL])]
    assert compute_signal(DIESEL, flat_market, news, articles, NOW).news_score == 0
    assert compute_signal(ProductKey.ETANOL, flat_market, news, articles, NOW).news_score > 0


@pytest.mark.parametrize("product", list(ProductKey))
def test_score_is_bounded(product):
    market = make_market(oil_daily=0.05, dollar_daily=0.02)
    articles = [article(f"Alta {i}") for i in range(20)]
    news = [
        assessment(i + 1, Direction.ALTA, category=c, products=list(ProductKey))
        for i, c in enumerate(list(Category) * 4)
    ][:20]
    signal = compute_signal(product, market, news, articles, NOW)
    assert -1 <= signal.score <= 1
