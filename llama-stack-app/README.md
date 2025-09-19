## Run Llama Stack Client

```bash
uv init --python 3.12
```

```bash
uv add llama-stack-client==0.2.21
```

```bash
uv run llama-stack-client configure --endpoint http://localhost:8321 --api-key none
```

```bash
uv run llama-stack-client models list
```

```bash
uv run llama-stack-client inference chat-completion --model-id "ollama/llama3.2:3b" --message "tell me a joke"
```

```bash
OTEL_SERVICE_NAME=client-app-service uv run script/main.py
```