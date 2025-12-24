## Llama Stack client

[/Users/fax/code/ai-play/llama-stack-03-app]

```bash
uv init
```

```bash
uv add llama-stack-client==0.3.4
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