"""Linha de comando: gera a recomendação de compra de combustível.

Exemplos:
    python -m fuel_agent
    python -m fuel_agent --produtos gasolina_comum,diesel_s10
    python -m fuel_agent --foco "O fim da desoneração vai afetar o etanol?"
"""

import argparse
import logging
import sys

from fuel_agent.agent import generate_recommendation
from fuel_agent.guardrails import GuardrailViolation
from fuel_agent.llm import MissingAPIKeyError
from fuel_agent.products import PRODUCTS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recomenda antecipar, adiar ou manter a compra de combustível."
    )
    parser.add_argument(
        "--produtos", help=f"lista separada por vírgula (padrão: todos): {', '.join(PRODUCTS)}"
    )
    parser.add_argument("--foco", help="pergunta ou foco opcional para a análise")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        report_path, audit = generate_recommendation(args.produtos, args.foco)
    except GuardrailViolation as exc:
        print(f"Pedido bloqueado pelos guardrails: {exc}", file=sys.stderr)
        return 2
    except MissingAPIKeyError as exc:
        print(f"Configuração incompleta: {exc} Veja o .env.example.", file=sys.stderr)
        return 3
    print(f"Relatório: {report_path}")
    print(f"Auditoria: {audit.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
