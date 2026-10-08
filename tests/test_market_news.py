from datetime import UTC, date, datetime

import pytest
from conftest import make_market

from fuel_agent.market import LITERS_PER_GALLON, build_indicator, parse_fred_csv, to_brl
from fuel_agent.news import parse_feed


def test_parse_fred_csv_skips_missing_values():
    text = "observation_date,DCOILBRENTEU\n2026-10-01,120.5\n2026-10-02,.\n2026-10-03,\n"
    assert parse_fred_csv(text) == {date(2026, 10, 1): 120.5}


def test_to_brl_uses_last_available_dollar():
    dollar = {date(2026, 10, 1): 5.0}
    usd = {date(2026, 10, 1): 2.0, date(2026, 10, 3): 3.0}  # dia 3 sem cotação do dólar
    assert to_brl(usd, dollar, 1 / LITERS_PER_GALLON) == pytest.approx(
        {date(2026, 10, 1): 10 / LITERS_PER_GALLON, date(2026, 10, 3): 15 / LITERS_PER_GALLON}
    )


def test_indicator_changes():
    series = {date(2026, 9, 1): 100.0, date(2026, 10, 1): 110.0, date(2026, 10, 8): 121.0}
    indicator = build_indicator("dolar", series)
    assert indicator.change_7d == pytest.approx(0.10)
    assert indicator.change_30d == pytest.approx(0.21)


def test_snapshot_converts_international_prices_to_reais():
    market = make_market()
    gasoline = market.indicators["gasolina_golfo_brl"]
    assert gasoline.unit == "R$/litro"
    assert gasoline.last_value == pytest.approx(2.5 * 5.0 / LITERS_PER_GALLON, rel=1e-3)


def test_parse_feed_filters_old_dedups_and_cleans():
    feed = """<?xml version="1.0"?><rss><channel>
      <item><title>Diesel sobe em Minas</title>
        <pubDate>Thu, 08 Oct 2026 12:00:00 +0000</pubDate>
        <description>&lt;p&gt;S10 subiu &lt;b&gt;44 centavos&lt;/b&gt;&lt;/p&gt;</description>
      </item>
      <item><title>Diesel sobe em Minas</title>
        <pubDate>Thu, 08 Oct 2026 11:00:00 +0000</pubDate></item>
      <item><title>Notícia antiga</title><pubDate>Mon, 01 Jun 2026 12:00:00 +0000</pubDate></item>
    </channel></rss>"""
    articles = parse_feed(
        feed,
        origin="minaspetro",
        default_source="MinasPetro",
        max_age_days=10,
        limit=10,
        now=datetime(2026, 10, 8, 20, tzinfo=UTC),
    )
    assert [a.title for a in articles] == ["Diesel sobe em Minas"]
    assert articles[0].summary == "S10 subiu 44 centavos"
    assert articles[0].source == "MinasPetro"
