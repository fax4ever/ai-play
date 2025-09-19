## Run Llama Stack Server

```bash
uv init --python 3.12
```

```bash
uv add llama-stack==0.2.21
```

```bash
uv run llama stack list
```

```bash
uv run llama stack build --distro starter --image-type venv
```

```text
You can find the newly-built distribution here: /home/fax/.llama/distributions/starter/starter-run.yaml
You can run the new Llama Stack distro via: llama stack run /home/fax/.llama/distributions/starter/starter-run.yaml --image-type venv
```

```bash
OLLAMA_URL=http://localhost:11434 uv run llama stack run starter --image-type venv
```

```bash
cp /Users/fax/.llama/distributions/starter/starter-run.yaml config/run.yaml
```

```bash
OLLAMA_URL=http://localhost:11434 OTEL_SERVICE_NAME=llama-fax OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 TELEMETRY_SINKS="console,sqlite,otel_trace" uv run llama stack run config/run.yaml --image-type venv
```