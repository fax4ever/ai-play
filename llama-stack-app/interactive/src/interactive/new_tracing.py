import httpx
import asyncio
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

def run1():
    url = "http://localhost:4318"
    HTTPXClientInstrumentor().instrument()

    with httpx.Client() as client:
        response = client.get(url)

    async def get(url):
        async with httpx.AsyncClient() as client:
            response = await client.get(url)

    asyncio.run(get(url))