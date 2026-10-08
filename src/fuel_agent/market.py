"""Indicadores de mercado: dólar (Banco Central) e petróleo/derivados (FRED), em reais.

Fontes públicas e sem chave:
- Dólar comercial (venda), série 1 do SGS do Banco Central.
- Brent (DCOILBRENTEU), gasolina convencional no Golfo dos EUA (DGASUSGULF) e diesel no
  Golfo dos EUA (DDFUELUSGULF), via FRED / EIA.

O Golfo dos EUA é a principal origem dos combustíveis importados pelo Brasil, por isso esses
preços convertidos em R$/litro indicam a pressão internacional sobre o preço da distribuidora.
"""

import csv
import io
import logging
from datetime import date, timedelta

import httpx
from pydantic import BaseModel

from fuel_agent.http import http_client

logger = logging.getLogger(__name__)

BCB_SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
LITERS_PER_GALLON = 3.785411784

FRED_SERIES = {
    "brent_usd": "DCOILBRENTEU",
    "gasolina_golfo_usd": "DGASUSGULF",
    "diesel_golfo_usd": "DDFUELUSGULF",
}

INDICATOR_LABELS = {
    "dolar": ("Dólar comercial", "R$/US$"),
    "brent_brl": ("Petróleo Brent", "R$/barril"),
    "gasolina_golfo_brl": ("Gasolina no Golfo dos EUA", "R$/litro"),
    "diesel_golfo_brl": ("Diesel no Golfo dos EUA", "R$/litro"),
}

Series = dict[date, float]


class Indicator(BaseModel):
    key: str
    name: str
    unit: str
    last_date: date
    last_value: float
    change_7d: float | None  # proporção (0.03 = +3%)
    change_30d: float | None


class MarketSnapshot(BaseModel):
    indicators: dict[str, Indicator]
    series: dict[str, dict[date, float]]  # séries em reais, para o gráfico


class MarketDataError(RuntimeError):
    pass


def fetch_dollar(start: date, end: date, *, client: httpx.Client) -> Series:
    response = client.get(
        BCB_SGS_URL,
        params={
            "formato": "json",
            "dataInicial": start.strftime("%d/%m/%Y"),
            "dataFinal": end.strftime("%d/%m/%Y"),
        },
    )
    response.raise_for_status()
    return {
        date(int(y), int(m), int(d)): float(row["valor"])
        for row in response.json()
        for d, m, y in [row["data"].split("/")]
    }


def parse_fred_csv(text: str) -> Series:
    """CSV do FRED (data, valor); dias sem negociação vêm como '.' ou vazio e são ignorados."""
    reader = csv.reader(io.StringIO(text))
    next(reader, None)  # cabeçalho
    series: Series = {}
    for row in reader:
        if len(row) < 2 or row[1] in ("", "."):
            continue
        series[date.fromisoformat(row[0])] = float(row[1])
    return series


def fetch_fred(series_id: str, start: date, *, client: httpx.Client) -> Series:
    response = client.get(FRED_CSV_URL, params={"id": series_id, "cosd": start.isoformat()})
    response.raise_for_status()
    return parse_fred_csv(response.text)


def _value_on_or_before(series: Series, day: date) -> float | None:
    candidates = [d for d in series if d <= day]
    return series[max(candidates)] if candidates else None


def to_brl(usd: Series, dollar: Series, factor: float = 1.0) -> Series:
    """Converte uma série em US$ para R$ usando o dólar do mesmo dia (ou o último disponível)."""
    converted = {}
    for day, value in usd.items():
        rate = _value_on_or_before(dollar, day)
        if rate is not None:
            converted[day] = value * rate * factor
    return converted


def _change(series: Series, last_day: date, days: int) -> float | None:
    previous = _value_on_or_before(series, last_day - timedelta(days=days))
    return series[last_day] / previous - 1 if previous else None


def build_indicator(key: str, series: Series) -> Indicator:
    if not series:
        raise MarketDataError(f"Série vazia: {key}")
    last_day = max(series)
    name, unit = INDICATOR_LABELS[key]
    return Indicator(
        key=key,
        name=name,
        unit=unit,
        last_date=last_day,
        last_value=round(series[last_day], 4),
        change_7d=_change(series, last_day, 7),
        change_30d=_change(series, last_day, 30),
    )


def build_snapshot(dollar: Series, usd_series: dict[str, Series]) -> MarketSnapshot:
    brl = {
        "dolar": dollar,
        "brent_brl": to_brl(usd_series["brent_usd"], dollar),
        "gasolina_golfo_brl": to_brl(
            usd_series["gasolina_golfo_usd"], dollar, 1 / LITERS_PER_GALLON
        ),
        "diesel_golfo_brl": to_brl(usd_series["diesel_golfo_usd"], dollar, 1 / LITERS_PER_GALLON),
    }
    return MarketSnapshot(
        indicators={key: build_indicator(key, series) for key, series in brl.items()},
        series=brl,
    )


def fetch_market(
    history_days: int, today: date | None = None, timeout: float = 30
) -> MarketSnapshot:
    today = today or date.today()
    start = today - timedelta(days=history_days)
    with http_client(timeout) as client:
        dollar = fetch_dollar(start, today, client=client)
        usd = {key: fetch_fred(sid, start, client=client) for key, sid in FRED_SERIES.items()}
    logger.info("Mercado: dólar até %s, Brent até %s", max(dollar), max(usd["brent_usd"]))
    return build_snapshot(dollar, usd)
