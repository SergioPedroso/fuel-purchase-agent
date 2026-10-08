"""Agente orquestrador em LangGraph.

Fluxo (o pedido já chega validado pelos guardrails de entrada em `generate_recommendation`):

    orquestrador <-> executar_tools
         |
    garantir_cobertura -> classificar_noticias -> calcular_sinais -> redigir_analise
                                                                          ^      |
                                                                 revisão  +-- validar_saida
                                                                                 |
                                                                          montar_relatorio

- `orquestrador`: o LLM decide quais tools chamar (mercado, temas de notícias, MinasPetro).
- `garantir_cobertura`: executa deterministicamente qualquer fonte obrigatória esquecida.
- `classificar_noticias`: o LLM classifica cada notícia (direção, intensidade, produtos).
- `calcular_sinais`: código determinístico combina mercado e notícias em uma decisão por produto.
- `redigir_analise`: o LLM explica as decisões (não pode alterá-las).
- `validar_saida`: confere números, coerência com a decisão, citações e dados pessoais.
"""

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from fuel_agent.analysis import PurchaseAnalysis
from fuel_agent.audit import AuditTrail
from fuel_agent.config import Settings, get_settings
from fuel_agent.guardrails import (
    GuardrailViolation,
    contains_injection,
    decimals_in,
    find_unverified_decimals,
    find_unverified_percentages,
    is_hedged,
    mask_pii,
    percentages_in,
    validate_request,
)
from fuel_agent.llm import LLMProvider, ToolCall, ToolOutcome, ToolSession, create_provider
from fuel_agent.news import NewsTopic
from fuel_agent.products import PRODUCTS
from fuel_agent.prompts import (
    CLASSIFIER_SYSTEM,
    CLASSIFIER_USER_TEMPLATE,
    FOCUS_BLOCK,
    ORCHESTRATOR_SYSTEM,
    REVISION_REQUEST,
    WRITER_SYSTEM,
    WRITER_USER_TEMPLATE,
)
from fuel_agent.report import render_report
from fuel_agent.signals import (
    Decision,
    NewsAssessment,
    NewsClassification,
    ProductSignal,
    compute_signal,
)
from fuel_agent.tools import (
    MINASPETRO,
    REQUIRED_NEWS_SOURCES,
    RunContext,
    execute_tool,
    market_payload,
    news_payload,
    tool_definitions,
)

MAX_WRITER_ATTEMPTS = 2

# Frases que, na justificativa, contradizem a decisão calculada.
CONTRADICTIONS = {
    Decision.ANTECIPAR: re.compile(r"\b(adi(ar|e)|esper(ar|e)) (a |as )?compras?\b", re.I),
    Decision.ADIAR: re.compile(r"\b(antecip(ar|e)|adiant(ar|e)) (a |as )?compras?\b", re.I),
}


class AgentState(TypedDict, total=False):
    pending_calls: list[ToolCall]
    steps: int
    assessments: list[NewsAssessment]
    signals: list[ProductSignal]
    analysis: PurchaseAnalysis
    writer_attempts: int
    validation_problems: list[str]
    warnings: list[str]
    report_path: Path


class FuelPurchaseAgent:
    def __init__(self, ctx: RunContext, llm: LLMProvider, settings: Settings):
        self.ctx = ctx
        self.llm = llm
        self.settings = settings
        self._session: ToolSession | None = None
        self.graph = self._build_graph()

    # ---- coleta -------------------------------------------------------------------------

    def orchestrate(self, state: AgentState) -> AgentState:
        if self._session is None:
            products = ", ".join(PRODUCTS[p].name for p in self.ctx.request.products)
            self._session = self.llm.start_tool_session(
                "orchestrator",
                ORCHESTRATOR_SYSTEM,
                f"Colete as evidências para a decisão de compra de: {products}.",
                tool_definitions(),
            )
        turn = self._session.next_turn()
        self.ctx.audit.record(
            "agent_decision",
            "orchestrator",
            tool_requests=[{"tool": c.name, "arguments": c.arguments} for c in turn.tool_calls],
            message=turn.text,
        )
        return {"pending_calls": turn.tool_calls, "steps": state.get("steps", 0) + 1}

    def run_tools(self, state: AgentState) -> AgentState:
        outcomes = []
        for call in state["pending_calls"]:
            content, is_error = execute_tool(self.ctx, call.name, call.arguments)
            outcomes.append(ToolOutcome(call=call, content=content, is_error=is_error))
        self._session.add_tool_results(outcomes)
        return {"pending_calls": []}

    def ensure_coverage(self, state: AgentState) -> AgentState:
        warnings = list(state.get("warnings", []))
        missing = []
        if self.ctx.market is None:
            missing.append(("consultar_mercado", {}))
        for source in REQUIRED_NEWS_SOURCES:
            if source in self.ctx.news_sources_done:
                continue
            if source == MINASPETRO:
                missing.append(("buscar_noticias_minaspetro", {}))
            else:
                missing.append(("buscar_noticias", {"tema": source}))
        for tool, args in missing:
            self.ctx.audit.record(
                "guardrail",
                "coverage",
                decision="forced_tool_execution",
                tool=tool,
                arguments=args,
                reason="fonte obrigatória não consultada pelo orquestrador",
            )
            _, is_error = execute_tool(self.ctx, tool, args)
            if is_error:
                warnings.append(f"Fonte indisponível: {tool} {args or ''}".strip())
        if self.ctx.market is None:
            raise RuntimeError("Dados de mercado indisponíveis; recomendação não pode ser gerada.")
        warnings.extend(_freshness_warnings(self.ctx))
        if not self.ctx.articles:
            warnings.append("Nenhuma notícia disponível: sinal baseado apenas no mercado.")
        return {"warnings": warnings}

    # ---- classificação e sinal ----------------------------------------------------------

    def classify_news(self, state: AgentState) -> AgentState:
        warnings = list(state.get("warnings", []))
        if not self.ctx.articles:
            return {"assessments": [], "warnings": warnings}
        system = CLASSIFIER_SYSTEM.format(
            horizon=self.settings.horizon_days, today=self.ctx.today.strftime("%d/%m/%Y")
        )
        payload = news_payload(list(enumerate(self.ctx.articles, start=1)))
        try:
            result = self.llm.structured(
                "news_classifier",
                system,
                CLASSIFIER_USER_TEMPLATE.format(news_json=json.dumps(payload, ensure_ascii=False)),
                NewsClassification,
            )
        except Exception as exc:  # sem classificação, o sinal usa só o mercado
            self.ctx.audit.record("guardrail", "classification", decision="failed", error=str(exc))
            warnings.append("Falha na classificação das notícias: sinal baseado só no mercado.")
            return {"assessments": [], "warnings": warnings}

        valid, seen, downgraded = [], set(), []
        for item in result.avaliacoes:
            if not 1 <= item.id <= len(self.ctx.articles) or item.id in seen:
                continue
            seen.add(item.id)
            if item.evento_confirmado and is_hedged(self.ctx.articles[item.id - 1].title):
                # "pode terminar", "governo avalia": possibilidade, não fato decidido.
                item = item.model_copy(update={"evento_confirmado": False})
                downgraded.append(item.id)
            valid.append(item)
        unclassified = sorted(set(range(1, len(self.ctx.articles) + 1)) - seen)
        self.ctx.audit.record(
            "guardrail",
            "classification",
            decision="accepted",
            classified=len(valid),
            discarded=len(result.avaliacoes) - len(valid),
            unclassified_ids=unclassified,
            downgraded_confirmed_ids=downgraded,
        )
        self.ctx.audit.record("news_assessments", "news_classifier", assessments=valid)
        return {"assessments": valid, "warnings": warnings}

    def compute_signals(self, state: AgentState) -> AgentState:
        now = datetime.combine(self.ctx.today, datetime.max.time(), tzinfo=UTC)
        signals = [
            compute_signal(p, self.ctx.market, state["assessments"], self.ctx.articles, now)
            for p in self.ctx.request.products
        ]
        self.ctx.audit.record("signals", "signal_engine", signals=signals)
        return {"signals": signals}

    # ---- redação e validação ------------------------------------------------------------

    def write_analysis(self, state: AgentState) -> AgentState:
        attempts = state.get("writer_attempts", 0) + 1
        user_content = self._writer_input(state["signals"])
        if state.get("validation_problems"):
            user_content += "\n<versao_anterior>\n" + state["analysis"].model_dump_json()
            user_content += "\n</versao_anterior>\n\n" + REVISION_REQUEST.format(
                problems="\n".join(f"- {p}" for p in state["validation_problems"])
            )
        analysis = self.llm.structured(
            f"writer_attempt_{attempts}", WRITER_SYSTEM, user_content, PurchaseAnalysis
        )
        return {"analysis": analysis, "writer_attempts": attempts}

    def validate_output(self, state: AgentState) -> AgentState:
        analysis: PurchaseAnalysis = state["analysis"]
        signals = {s.product: s for s in state["signals"]}
        text = analysis.full_text()
        problems = []

        percents, decimals = self._allowed_numbers(state["signals"])
        if wrong := find_unverified_percentages(text, percents):
            problems.append("Percentuais sem correspondência nos dados: " + ", ".join(wrong))
        if wrong := find_unverified_decimals(text, decimals):
            problems.append("Valores sem correspondência nos dados: " + ", ".join(wrong))

        covered = {p.produto for p in analysis.por_produto}
        if missing := set(signals) - covered:
            problems.append(f"Produtos sem justificativa: {sorted(missing)}")
        for rationale in analysis.por_produto:
            signal = signals.get(rationale.produto)
            pattern = CONTRADICTIONS.get(signal.decision) if signal else None
            if pattern and pattern.search(rationale.justificativa):
                problems.append(
                    f"A justificativa de {rationale.produto} contradiz a decisão "
                    f"{signal.decision} calculada."
                )

        if invalid := [
            i for i in analysis.noticias_citadas if not 1 <= i <= len(self.ctx.articles)
        ]:
            problems.append(f"ids de notícias inexistentes citados: {invalid}")
        if contains_injection(text):
            problems.append("O texto contém instruções ou trechos de prompt.")

        self.ctx.audit.record(
            "guardrail",
            "output_validation",
            decision="approved" if not problems else "rejected",
            attempt=state["writer_attempts"],
            problems=problems,
            pii_masked=mask_pii(text)[1],
        )
        warnings = list(state.get("warnings", []))
        if problems and state["writer_attempts"] >= MAX_WRITER_ATTEMPTS:
            warnings.extend(f"Validação automática: {p}" for p in problems)
            problems = []  # segue com aviso explícito no relatório
        return {
            "analysis": analysis.masked(),
            "validation_problems": problems,
            "warnings": warnings,
        }

    def render(self, state: AgentState) -> AgentState:
        path = render_report(
            self.ctx,
            state["signals"],
            state["assessments"],
            state["analysis"],
            state.get("warnings", []),
        )
        self.ctx.audit.record("report_rendered", "renderer", path=path)
        return {"report_path": path}

    # ---- roteamento ---------------------------------------------------------------------

    def _after_orchestrator(self, state: AgentState) -> str:
        if state["pending_calls"] and state["steps"] < self.settings.max_agent_steps:
            return "executar_tools"
        if state["pending_calls"]:
            self.ctx.audit.record(
                "guardrail", "max_steps", decision="stop_loop", steps=state["steps"]
            )
        return "garantir_cobertura"

    def _after_validation(self, state: AgentState) -> str:
        return "redigir_analise" if state["validation_problems"] else "montar_relatorio"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("orquestrador", self.orchestrate)
        graph.add_node("executar_tools", self.run_tools)
        graph.add_node("garantir_cobertura", self.ensure_coverage)
        graph.add_node("classificar_noticias", self.classify_news)
        graph.add_node("calcular_sinais", self.compute_signals)
        graph.add_node("redigir_analise", self.write_analysis)
        graph.add_node("validar_saida", self.validate_output)
        graph.add_node("montar_relatorio", self.render)

        graph.add_edge(START, "orquestrador")
        graph.add_conditional_edges("orquestrador", self._after_orchestrator)
        graph.add_edge("executar_tools", "orquestrador")
        graph.add_edge("garantir_cobertura", "classificar_noticias")
        graph.add_edge("classificar_noticias", "calcular_sinais")
        graph.add_edge("calcular_sinais", "redigir_analise")
        graph.add_edge("redigir_analise", "validar_saida")
        graph.add_conditional_edges("validar_saida", self._after_validation)
        graph.add_edge("montar_relatorio", END)
        return graph.compile()

    # ---- apoio --------------------------------------------------------------------------

    def _writer_input(self, signals: list[ProductSignal]) -> str:
        signals_payload = [
            {
                "produto": s.product.value,
                "nome": PRODUCTS[s.product].name,
                "decisao": s.decision.value,
                "confianca": s.confidence.value,
                "pontuacao": s.score,
                "pontuacao_mercado": s.market_score,
                "pontuacao_noticias": s.news_score,
                "evento_confirmado": s.confirmed_event,
                "sinais_conflitantes": s.conflicting_signals,
                "principais_fatores": [d.description for d in s.drivers],
            }
            for s in signals
        ]
        focus = self.ctx.request.focus
        return WRITER_USER_TEMPLATE.format(
            today=self.ctx.today.strftime("%d/%m/%Y"),
            horizon=self.settings.horizon_days,
            products=", ".join(PRODUCTS[p].name for p in self.ctx.request.products),
            focus_block=FOCUS_BLOCK.format(focus=focus) if focus else "",
            signals_json=json.dumps(signals_payload, ensure_ascii=False, indent=1),
            market_json=json.dumps(market_payload(self.ctx.market), ensure_ascii=False, indent=1),
            news_json=json.dumps(
                news_payload(list(enumerate(self.ctx.articles, start=1))), ensure_ascii=False
            ),
        )

    def _allowed_numbers(self, signals: list[ProductSignal]) -> tuple[set[float], set[float]]:
        """Percentuais e valores decimais que o texto pode citar."""
        percents: set[float] = set()
        decimals: set[float] = set()
        for ind in self.ctx.market.indicators.values():
            decimals |= {round(ind.last_value, 2), round(ind.last_value, 4)}
            for change in (ind.change_7d, ind.change_30d):
                if change is not None:
                    percents.add(round(abs(change) * 100, 1))
        for s in signals:
            decimals |= {abs(s.score), abs(s.market_score), abs(s.news_score)}
            for d in s.drivers:
                percents |= percentages_in(d.description)
        for article in self.ctx.articles:
            text = f"{article.title} {article.summary}"
            percents |= percentages_in(text)
            decimals |= decimals_in(text)
        return percents, decimals


def _freshness_warnings(ctx: RunContext) -> list[str]:
    """Avisa se algum indicador está atrasado (feriado, fonte sem atualização)."""
    stale = [
        f"{ind.name} (último dado em {ind.last_date:%d/%m/%Y})"
        for ind in ctx.market.indicators.values()
        if (ctx.today - ind.last_date).days > 5
    ]
    return [f"Indicadores desatualizados: {', '.join(stale)}."] if stale else []


def generate_recommendation(
    products: str | None = None,
    focus: str | None = None,
    *,
    settings: Settings | None = None,
    llm_client=None,
    today: date | None = None,
) -> tuple[Path, AuditTrail]:
    """Ponto de entrada: valida o pedido, executa o grafo e devolve o relatório e a auditoria."""
    settings = settings or get_settings()
    audit = AuditTrail(settings.logs_dir)
    try:
        request = validate_request(products, focus)
    except GuardrailViolation as exc:
        audit.record("guardrail", "input_validation", decision="rejected", reason=str(exc))
        raise
    today = today or date.today()
    audit.record(
        "run_started",
        "cli",
        request=request,
        today=today,
        horizon_days=settings.horizon_days,
        provider=settings.llm_provider,
        model=settings.model_name,
        news_topics=[t.value for t in NewsTopic],
    )

    llm = create_provider(settings, audit, llm_client)
    ctx = RunContext(
        request=request,
        audit=audit,
        output_dir=settings.outputs_dir / audit.run_id,
        today=today,
        horizon_days=settings.horizon_days,
        market_history_days=settings.market_history_days,
        news_max_age_days=settings.news_max_age_days,
        news_per_query=settings.news_per_query,
        model_label=f"{settings.llm_provider} / {settings.model_name}",
    )
    agent = FuelPurchaseAgent(ctx, llm, settings)
    try:
        final_state = agent.graph.invoke({}, {"recursion_limit": 4 * settings.max_agent_steps})
    except Exception as exc:
        audit.record("run_failed", "agent", error=f"{type(exc).__name__}: {exc}")
        raise
    audit.record("run_finished", "agent", report=final_state["report_path"])
    return final_state["report_path"], audit
