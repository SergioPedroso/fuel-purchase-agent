"""Cliente HTTP compartilhado: identifica o projeto no User-Agent e repete falhas de conexão."""

import httpx

USER_AGENT = "fuel-purchase-agent/0.1 (+https://github.com/SergioPedroso/fuel-purchase-agent)"


def http_client(timeout: float = 30) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        transport=httpx.HTTPTransport(retries=2),
    )
