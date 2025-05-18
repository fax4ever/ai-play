from llama_stack_client import LlamaStackClient
from fastapi import FastAPI


app = FastAPI()


@app.get("/")
async def root():
    client = LlamaStackClient(base_url="http://localhost:8321")
    models = client.models.list()
    return {"models": str(models)}
