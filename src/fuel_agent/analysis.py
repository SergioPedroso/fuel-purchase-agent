"""Formato estruturado do texto escrito pelo LLM para o relatório de compra."""

from pydantic import BaseModel, Field

from fuel_agent.guardrails import mask_pii
from fuel_agent.products import ProductKey


class ProductRationale(BaseModel):
    produto: ProductKey
    justificativa: str = Field(description="2 a 3 frases explicando a decisão já calculada")


class PurchaseAnalysis(BaseModel):
    resumo_executivo: str = Field(description="3 a 5 frases para o comprador")
    por_produto: list[ProductRationale]
    contexto_mercado: str = Field(description="Leitura do dólar, petróleo e derivados")
    riscos: list[str] = Field(description="O que poderia mudar a recomendação nos próximos dias")
    noticias_citadas: list[int] = Field(description="ids das notícias usadas")

    def full_text(self) -> str:
        return "\n".join(
            [
                self.resumo_executivo,
                self.contexto_mercado,
                *(p.justificativa for p in self.por_produto),
                *self.riscos,
            ]
        )

    def masked(self) -> "PurchaseAnalysis":
        def mask(text: str) -> str:
            return mask_pii(text)[0]

        return self.model_copy(
            update={
                "resumo_executivo": mask(self.resumo_executivo),
                "contexto_mercado": mask(self.contexto_mercado),
                "por_produto": [
                    p.model_copy(update={"justificativa": mask(p.justificativa)})
                    for p in self.por_produto
                ],
                "riscos": [mask(r) for r in self.riscos],
            }
        )
