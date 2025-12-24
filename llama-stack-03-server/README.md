## Llama Stack server

```bash
uv init
```

```bash
uv add llama-stack==0.3.2
```

```bash
uv run llama stack list-deps starter | xargs -L1 uv pip install
```

See the current distributions:
```bash
uv run llama stack list
```

Clean up possible old distribution:
```bash
uv run llama stack rm starter
```

Run:
```bash
OLLAMA_URL=http://localhost:11434 uv run llama stack run starter
```

Run:
```bash
OLLAMA_URL=http://localhost:11434 uv run llama stack run config/run.yaml 
```

Run:
```bash
OLLAMA_URL=http://localhost:11434 OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 uv run llama stack run config/run.yaml 
```

Run:
```bash
OLLAMA_URL=http://localhost:11434 OTEL_SERVICE_NAME=llama-fax OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 uv run llama stack run config/run.yaml 
```

Run: (advanced options)
```bash
OLLAMA_URL=http://localhost:11434 OTEL_SERVICE_NAME=llama-fax OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 TELEMETRY_SINKS="['otel_trace']" uv run llama stack run config/run.yaml
```