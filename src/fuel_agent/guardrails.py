"""Guardrails determinísticos aplicados antes e depois do LLM.

1. Entrada: produtos validados contra lista fechada; foco opcional limitado, restrito ao tema
   e checado contra injeção de instruções e dados pessoais.
2. Tools: argumentos são enumerações fechadas (tema de notícia), nunca texto livre.
3. Conteúdo externo: notícias com padrões de injeção são descartadas; dados pessoais mascarados.
4. Saída: percentuais e valores citados pelo LLM precisam bater com os indicadores ou com as
   fontes; a justificativa não pode contradizer a decisão calculada.
"""

import re
import unicodedata

from pydantic import BaseModel, Field, ValidationError, field_validator

from fuel_agent.news import NewsArticle
from fuel_agent.products import ProductKey, parse_products

MAX_FOCUS_CHARS = 300
PERCENT_TOLERANCE_PP = 0.15
DECIMAL_TOLERANCE = 0.011

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"ignore (as )?instru[cç][oõ]es",
    r"desconsidere (as |todas as )?(instru[cç][oõ]es|regras)",
    r"system prompt|prompt do sistema",
    r"you are now|voc[eê] agora [eé]",
    r"act as|aja como|finja (ser|que)",
    r"</?(system|assistant|user|instructions?)>",
    r"reveal (your|the) (instructions|prompt)|revele (suas|as) instru",
    r"jailbreak|developer mode|modo desenvolvedor",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

TOPIC_KEYWORDS = [
    "combust", "gasolina", "etanol", "diesel", "petroleo", "brent", "dolar", "cambio",
    "petrobras", "preco", "reajuste", "imposto", "icms", "pis", "cofins", "cide",
    "desoneracao", "distribuidora", "betim", "paulinia", "guerra", "opep", "compra",
    "estoque", "safra", "usina", "v-power", "shell", "raizen",
]  # fmt: skip

PII_PATTERNS = {
    "CPF": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    "CNPJ": re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "TELEFONE": re.compile(r"(?<!\d)\(?\d{2}\)?\s?9?\d{4}-\d{4}(?!\d)"),
}

PERCENT_RE = re.compile(r"(-?\d{1,3}(?:[.,]\d+)?)\s?%")
# Valores com casas decimais (R$ 5,01; 4,68): os que mais importam numa decisão de compra.
DECIMAL_RE = re.compile(r"(?<![\d.,])(\d{1,4},\d{1,4})(?![\d%]|\s?%)")


# Linguagem de possibilidade: notícia com esses termos no título não é fato confirmado.
HEDGE_RE = re.compile(
    r"\b(pode|podem|poder[aá]|poderiam|deve|devem|dever[aá]|avalia|avaliam|estuda|estudam|"
    r"pretende|prev[eê]|possivel|possível|proposta|projeto|cogita|sinaliza|tende)\b",
    re.IGNORECASE,
)


def is_hedged(title: str) -> bool:
    """True se o título fala de possibilidade (\"pode terminar\", \"governo avalia\")."""
    return bool(HEDGE_RE.search(title))


class GuardrailViolation(ValueError):
    """Pedido bloqueado por um guardrail de entrada."""


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def contains_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(_normalize(text)) or _INJECTION_RE.search(text))


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for label, pattern in PII_PATTERNS.items():
        text, n = pattern.subn(f"[{label} REMOVIDO]", text)
        if n:
            counts[label] = n
    return text, counts


class PurchaseRequest(BaseModel):
    products: list[ProductKey]
    focus: str | None = Field(default=None, description="Pergunta ou foco opcional")

    @field_validator("focus")
    @classmethod
    def _validate_focus(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip()
        if len(value) > MAX_FOCUS_CHARS:
            raise ValueError(f"O foco deve ter no máximo {MAX_FOCUS_CHARS} caracteres.")
        if contains_injection(value):
            raise ValueError("O foco contém instruções não permitidas.")
        if not any(k in _normalize(value) for k in TOPIC_KEYWORDS):
            raise ValueError("O foco precisa estar relacionado a combustíveis e à compra.")
        if mask_pii(value)[1]:
            raise ValueError("O foco não pode conter dados pessoais.")
        return value


def validate_request(products: str | None, focus: str | None) -> PurchaseRequest:
    """Cria o pedido validado; qualquer violação vira `GuardrailViolation` com motivo legível."""
    try:
        return PurchaseRequest(products=parse_products(products), focus=focus)
    except ValidationError as exc:
        reasons = "; ".join(str(e.get("ctx", {}).get("error", e["msg"])) for e in exc.errors())
        raise GuardrailViolation(reasons) from exc
    except ValueError as exc:
        raise GuardrailViolation(str(exc)) from exc


class NewsScreening(BaseModel):
    accepted: list[NewsArticle]
    rejected: list[dict]


def screen_news(articles: list[NewsArticle]) -> NewsScreening:
    """Descarta notícias com tentativa de injeção e mascara dados pessoais nas demais."""
    accepted, rejected = [], []
    for article in articles:
        if contains_injection(f"{article.title} {article.summary}"):
            rejected.append({"title": article.title, "reason": "possível injeção de instruções"})
            continue
        accepted.append(
            article.model_copy(
                update={
                    "title": mask_pii(article.title)[0],
                    "summary": mask_pii(article.summary)[0],
                }
            )
        )
    return NewsScreening(accepted=accepted, rejected=rejected)


def _to_float(raw: str) -> float:
    return float(raw.replace(",", "."))


def percentages_in(text: str) -> set[float]:
    return {abs(_to_float(m)) for m in PERCENT_RE.findall(text)}


def decimals_in(text: str) -> set[float]:
    return {_to_float(m) for m in DECIMAL_RE.findall(text)}


def find_unverified_percentages(text: str, allowed: set[float]) -> list[str]:
    return [
        f"{raw}%"
        for raw in PERCENT_RE.findall(text)
        if not any(abs(abs(_to_float(raw)) - a) <= PERCENT_TOLERANCE_PP for a in allowed)
    ]


def find_unverified_decimals(text: str, allowed: set[float]) -> list[str]:
    """Valores decimais (ex.: R$ 5,01) sem correspondência nos indicadores ou nas fontes."""
    return [
        raw
        for raw in DECIMAL_RE.findall(text)
        if not any(abs(_to_float(raw) - a) <= DECIMAL_TOLERANCE for a in allowed)
    ]
