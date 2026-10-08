"""Produtos analisados e o que move o preço de cada um na distribuidora.

Cada produto tem pesos para os indicadores de mercado e uma `market_share`: quanto do sinal
vem do mercado internacional versus das notícias. O etanol é formado sobretudo no mercado
interno (safra, usinas, Paulínia), então depende mais das notícias que do petróleo.
"""

from dataclasses import dataclass
from enum import StrEnum


class ProductKey(StrEnum):
    GASOLINA_COMUM = "gasolina_comum"
    GASOLINA_VPOWER = "gasolina_vpower"
    ETANOL = "etanol"
    ETANOL_VPOWER = "etanol_vpower"
    DIESEL_S10 = "diesel_s10"


@dataclass(frozen=True)
class Product:
    key: ProductKey
    name: str
    market_weights: dict[str, float]  # indicador -> peso (soma 1)
    market_share: float  # participação do mercado no sinal final (o resto vem das notícias)
    note: str = ""


_GASOLINE_MARKET = {"gasolina_golfo_brl": 0.6, "brent_brl": 0.2, "dolar": 0.2}
_ETHANOL_MARKET = {"gasolina_golfo_brl": 0.5, "dolar": 0.5}
_DIESEL_MARKET = {"diesel_golfo_brl": 0.6, "brent_brl": 0.2, "dolar": 0.2}

PRODUCTS: dict[ProductKey, Product] = {
    p.key: p
    for p in (
        Product(ProductKey.GASOLINA_COMUM, "Gasolina comum", _GASOLINE_MARKET, 0.4),
        Product(
            ProductKey.GASOLINA_VPOWER,
            "Gasolina Shell V-Power",
            _GASOLINE_MARKET,
            0.4,
            note="Acompanha a gasolina comum (mesma base, com aditivos).",
        ),
        Product(ProductKey.ETANOL, "Etanol hidratado", _ETHANOL_MARKET, 0.1),
        Product(
            ProductKey.ETANOL_VPOWER,
            "Etanol Shell V-Power",
            _ETHANOL_MARKET,
            0.1,
            note="Acompanha o etanol hidratado (mesma base, com aditivos).",
        ),
        Product(ProductKey.DIESEL_S10, "Diesel S10", _DIESEL_MARKET, 0.5),
    )
}


def parse_products(raw: str | None) -> list[ProductKey]:
    """'gasolina_comum,diesel_s10' -> [ProductKey, ...]; vazio = todos; fora da lista = erro."""
    if not raw or not raw.strip():
        return list(PRODUCTS)
    keys = []
    for item in raw.split(","):
        value = item.strip().lower()
        if value not in PRODUCTS:
            valid = ", ".join(PRODUCTS)
            raise ValueError(f"Produto inválido: {item!r}. Use: {valid}.")
        keys.append(ProductKey(value))
    return list(dict.fromkeys(keys))
