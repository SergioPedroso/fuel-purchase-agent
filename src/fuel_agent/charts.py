"""Gráficos do relatório (PNG estático via matplotlib).

1. Indicadores de mercado em base 100: uma escala só, para comparar dólar, Brent e derivados
   em reais (unidades diferentes não dividem eixo).
2. Pontuação por produto: barras coloridas pela decisão final (vermelho = antecipar,
   azul = adiar, cinza = manter), com as faixas de decisão marcadas.
"""

from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from fuel_agent.market import INDICATOR_LABELS, MarketSnapshot  # noqa: E402
from fuel_agent.products import PRODUCTS  # noqa: E402
from fuel_agent.signals import DECISION_THRESHOLD, Decision, ProductSignal  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_MUTED = "#898781"
GRID = "#e4e3df"
SERIES_COLORS = {
    "dolar": "#2a78d6",
    "brent_brl": "#eb6834",
    "gasolina_golfo_brl": "#1baf7a",
    "diesel_golfo_brl": "#4a3aa7",
}
DECISION_COLORS = {
    Decision.ANTECIPAR: "#d03b3b",  # tende a subir
    Decision.ADIAR: "#2a78d6",  # tende a cair
    Decision.MANTER: "#c3c2b7",
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9, length=0)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _titles(fig, ax, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.06, ha="left", fontsize=13, fontweight="bold", color=TEXT_PRIMARY)
    ax.set_title(subtitle, loc="left", fontsize=9, color=TEXT_MUTED, pad=10)


def plot_market_index(market: MarketSnapshot, output_path: Path, days: int = 90) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for key, color in SERIES_COLORS.items():
        series = market.series[key]
        start = max(series) - timedelta(days=days)
        points = sorted((d, v) for d, v in series.items() if d >= start)
        base = points[0][1]
        xs = [d for d, _ in points]
        ys = [v / base * 100 for _, v in points]
        ax.plot(xs, ys, color=color, linewidth=2, label=INDICATOR_LABELS[key][0])
        ax.annotate(
            f"{ys[-1]:.0f}",
            (xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT_PRIMARY,
        )
    ax.axhline(100, color=TEXT_MUTED, linewidth=0.8, linestyle=":")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY, ncol=2)
    _titles(
        fig,
        ax,
        f"Mercado em reais — últimos {days} dias (base 100)",
        "Dólar (BCB); Brent, gasolina e diesel no Golfo dos EUA (FRED) convertidos para R$.",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)
    return output_path


def plot_signals(signals: list[ProductSignal], today: date, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 0.7 * len(signals) + 1.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(False)
    names = [PRODUCTS[s.product].name for s in reversed(signals)]
    scores = [s.score for s in reversed(signals)]
    # A cor segue a decisão final (que considera conflito de sinais), não só a pontuação.
    colors = [DECISION_COLORS[s.decision] for s in reversed(signals)]
    ax.barh(names, scores, color=colors, height=0.55)
    for y, (score, signal) in enumerate(zip(scores, reversed(signals), strict=True)):
        ax.annotate(
            f"{signal.decision.value} ({signal.confidence.value})",
            (score, y),
            xytext=(8 if score >= 0 else -8, 0),
            textcoords="offset points",
            ha="left" if score >= 0 else "right",
            va="center",
            fontsize=9,
            color=TEXT_PRIMARY,
        )
    for x in (-DECISION_THRESHOLD, DECISION_THRESHOLD):
        ax.axvline(x, color=TEXT_MUTED, linewidth=0.8, linestyle="--")
    ax.axvline(0, color=TEXT_MUTED, linewidth=1)
    ax.set_xlim(-1.35, 1.35)
    ax.set_xticks([-1, -DECISION_THRESHOLD, 0, DECISION_THRESHOLD, 1])
    ax.set_xticklabels(["cair", "-0,2", "0", "+0,2", "subir"])
    _titles(
        fig,
        ax,
        f"Tendência do preço na distribuidora — {today:%d/%m/%Y}",
        "Pontuação de -1 (tende a cair: adiar) a +1 (tende a subir: antecipar). "
        "Entre as linhas tracejadas: manter rotina.",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)
    return output_path
