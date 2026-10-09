"""Montagem do relatório de compra (Markdown + HTML).

Decisões, pontuações, indicadores, gráficos e a tabela de notícias vêm dos dados
determinísticos; o LLM contribui só com os textos explicativos (`PurchaseAnalysis`).
"""

from datetime import UTC, datetime
from pathlib import Path

import markdown

from fuel_agent.analysis import PurchaseAnalysis
from fuel_agent.charts import plot_market_index, plot_signals
from fuel_agent.products import PRODUCTS
from fuel_agent.signals import Decision, NewsAssessment, ProductSignal
from fuel_agent.tools import RunContext, format_number, format_percent

DISCLAIMER = (
    "Recomendação gerada automaticamente por um agente de IA a partir de indicadores públicos "
    "(Banco Central, FRED) e de notícias (Google News, MinasPetro). A decisão é calculada por "
    "regras determinísticas; os textos são produzidos por um LLM e validados automaticamente. "
    "É um apoio à decisão de compra, não uma garantia de preço."
)

DECISION_ICON = {Decision.ANTECIPAR: "🔺", Decision.ADIAR: "🔻", Decision.MANTER: "➖"}
DIRECTION_LABEL = {"alta": "↑ alta", "baixa": "↓ baixa", "neutra": "→ neutra"}

HTML_TEMPLATE = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; color: #0b0b0b;
         background: #f9f9f7; max-width: 980px; margin: 0 auto; padding: 24px 16px;
         line-height: 1.55; }}
  h1 {{ font-size: 1.7rem; margin-bottom: .2rem; }}
  h2 {{ margin-top: 2rem; border-bottom: 1px solid #e4e3df; padding-bottom: .3rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .93rem; display: block;
          overflow-x: auto; }}
  th, td {{ text-align: left; padding: 7px 9px; border-bottom: 1px solid #e4e3df;
           vertical-align: top; }}
  th {{ color: #52514e; font-weight: 600; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #e4e3df; border-radius: 6px; }}
  blockquote {{ margin: 1rem 0; padding: .6rem 1rem; background: #fff7e6;
               border-left: 4px solid #eda100; }}
  .meta, small {{ color: #52514e; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def practical_action(decision: Decision, purchases_per_week: int) -> str:
    """Tradução da decisão para o ritmo de compras do posto."""
    interval = round(7 / max(purchases_per_week, 1))
    if decision == Decision.ANTECIPAR:
        return (
            "Antecipe a próxima compra ou aumente o volume desta para cobrir também a "
            "seguinte, se houver espaço no tanque."
        )
    if decision == Decision.ADIAR:
        return (
            f"Compre só o necessário até a próxima compra (~{interval} dias) e reavalie antes "
            "dela; evite estoque extra."
        )
    return f"Siga a rotina ({purchases_per_week} compras por semana) e reavalie na próxima."


def _decision_table(signals: list[ProductSignal], purchases_per_week: int) -> str:
    rows = [
        "| Produto | Decisão | Confiança | Na prática | Pontuação (mercado / notícias) |",
        "|---|---|---|---|---|",
    ]
    for s in signals:
        score = f"{s.score:+.2f} ({s.market_score:+.2f} / {s.news_score:+.2f})".replace(".", ",")
        rows.append(
            f"| {PRODUCTS[s.product].name} | {DECISION_ICON[s.decision]} **{s.decision.value}** "
            f"| {s.confidence.value} | {practical_action(s.decision, purchases_per_week)} "
            f"| {score} |"
        )
    return "\n".join(rows)


def _product_sections(signals: list[ProductSignal], analysis: PurchaseAnalysis) -> str:
    rationale = {p.produto: p.justificativa for p in analysis.por_produto}
    sections = []
    for s in signals:
        product = PRODUCTS[s.product]
        drivers = "\n".join(
            f"- {'📈' if d.contribution > 0 else '📉'} {d.description}" for d in s.drivers
        )
        event = f"\n\n**Evento confirmado:** {s.confirmed_event}" if s.confirmed_event else ""
        if s.conflicting_signals:
            event += (
                "\n\n**Sinais conflitantes:** mercado e notícias apontam para lados opostos; "
                "por isso a recomendação é manter a rotina."
            )
        note = f"\n\n<small>{product.note}</small>" if product.note else ""
        sections.append(
            f"### {DECISION_ICON[s.decision]} {product.name}: {s.decision.value} "
            f"(confiança {s.confidence.value})\n\n"
            f"{rationale.get(s.product, '_Sem justificativa gerada._')}{event}\n\n"
            f"<small>Principais fatores:</small>\n\n{drivers}{note}\n"
        )
    return "\n".join(sections)


def _market_table(ctx: RunContext) -> str:
    rows = ["| Indicador | Último valor | Data | 7 dias | 30 dias |", "|---|---|---|---|---|"]
    for ind in ctx.market.indicators.values():
        rows.append(
            f"| {ind.name} | {format_number(ind.last_value)} {ind.unit} | "
            f"{ind.last_date:%d/%m/%Y} | {format_percent(ind.change_7d)} | "
            f"{format_percent(ind.change_30d)} |"
        )
    return "\n".join(rows)


def _news_table(ctx: RunContext, assessments: list[NewsAssessment]) -> str:
    by_id = {a.id: a for a in assessments}
    rows = ["| # | Data | Notícia | Efeito | Produtos |", "|---|---|---|---|---|"]
    relevant = 0
    for i, article in enumerate(ctx.articles, start=1):
        item = by_id.get(i)
        if not item or not item.relevante:
            continue
        relevant += 1
        when = f"{article.published_at:%d/%m}" if article.published_at else "s/d"
        effect = f"{DIRECTION_LABEL[item.direcao]} ({item.intensidade})"
        if item.evento_confirmado:
            effect += " · confirmado"
        products = ", ".join(PRODUCTS[p].name for p in item.produtos)
        title = article.title.replace("|", "/")
        rows.append(
            f"| {i} | {when} | [{title}]({article.url}) — {article.source}"
            f"<br><small>{item.motivo}</small> | {effect} | {products} |"
        )
    ignored = len(ctx.articles) - relevant
    rejected = (
        f" e {len(ctx.rejected_news)} descartadas pelos guardrails" if ctx.rejected_news else ""
    )
    footer = (
        f"\n\n<small>{relevant} notícias relevantes de {len(ctx.articles)} lidas; "
        f"{ignored} classificadas como sem efeito no preço de compra{rejected}.</small>"
    )
    return ("\n".join(rows) if relevant else "_Nenhuma notícia relevante no período._") + footer


def build_markdown(
    ctx: RunContext,
    signals: list[ProductSignal],
    assessments: list[NewsAssessment],
    analysis: PurchaseAnalysis,
    warnings: list[str],
    signal_chart: Path,
    market_chart: Path,
) -> str:
    generated = datetime.now(UTC).strftime("%d/%m/%Y %H:%M UTC")
    warning_block = "\n".join(f"> ⚠️ {w}" for w in warnings)
    risks = "\n".join(f"- {r}" for r in analysis.riscos)
    return f"""# Recomendação de compra de combustível — {ctx.today:%d/%m/%Y}

<p class="meta">Posto Shell · Lavras (MG) · bases de Betim e Paulínia · horizonte de
{ctx.horizon_days} dias · gerado em {generated} · execução
<code>{ctx.audit.run_id}</code></p>

{warning_block}

## Decisão por produto

{_decision_table(signals, ctx.purchases_per_week)}

![Tendência do preço por produto]({signal_chart.name})

## Resumo

{analysis.resumo_executivo}

## Justificativa por produto

{_product_sections(signals, analysis)}

## Mercado

{_market_table(ctx)}

![Indicadores de mercado em base 100]({market_chart.name})

{analysis.contexto_mercado}

## Notícias consideradas

{_news_table(ctx, assessments)}

## O que pode mudar a recomendação

{risks}

## Como a decisão é tomada

- **Mercado:** variação em 7 e 30 dias do dólar e dos preços internacionais em reais (Brent,
  gasolina e diesel no Golfo dos EUA, principal origem das importações). Peso maior no diesel
  e na gasolina; menor no etanol, formado sobretudo no mercado interno.
- **Notícias:** o LLM classifica cada notícia (direção, intensidade, produtos afetados, se é
  fato confirmado com data). O código soma os efeitos, com peso maior para notícias recentes.
- **Eventos confirmados** dentro do horizonte (reajuste anunciado, imposto com data) definem a
  direção, com confiança alta.
- **Decisão:** pontuação ≥ +0,2 → ANTECIPAR; ≤ −0,2 → ADIAR; entre as duas → MANTER ROTINA.
  Se mercado e notícias apontam forte para lados opostos, a decisão é MANTER ROTINA
  (antecipar ou adiar tem custo e não deve se basear em evidências contraditórias).
- **Modelo:** `{ctx.model_label}`. Auditoria completa em `logs/audit/{ctx.audit.run_id}.jsonl`.

---

<small>{DISCLAIMER}</small>
"""


def render_report(
    ctx: RunContext,
    signals: list[ProductSignal],
    assessments: list[NewsAssessment],
    analysis: PurchaseAnalysis,
    warnings: list[str],
) -> Path:
    """Grava gráficos, `relatorio.md` e `relatorio.html`; retorna o caminho do HTML."""
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    signal_chart = plot_signals(signals, ctx.today, ctx.output_dir / "sinal_por_produto.png")
    market_chart = plot_market_index(ctx.market, ctx.output_dir / "mercado_base100.png")
    content = build_markdown(
        ctx, signals, assessments, analysis, warnings, signal_chart, market_chart
    )
    (ctx.output_dir / "relatorio.md").write_text(content, encoding="utf-8")
    body = markdown.markdown(content, extensions=["tables"])
    html_path = ctx.output_dir / "relatorio.html"
    html_path.write_text(
        HTML_TEMPLATE.format(title=f"Compra de combustível — {ctx.today:%d/%m/%Y}", body=body),
        encoding="utf-8",
    )
    return html_path
