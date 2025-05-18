# Container DEMO

0. *Optionally* cleanup the podman environment

```shell
podman rm -fv $(podman ps -a -q)
```

Or simply remove the llama-stack container:

```shell
podman rm -fv --depend llamastack
```

1. Run Ollama service

```shell
OLLAMA_HOST=0.0.0.0 ollama serve
```

2. *Optionally* install the model

```shell
ollama pull llama3.2:3b-instruct-fp16
```

3. Run Llama stack service

```shell
podman run --platform linux/amd64 -it -v llamastack:/root/.llama \
    -p 8321:8321 \
    --env INFERENCE_MODEL="meta-llama/Llama-3.2-3B-Instruct" \
    --env OLLAMA_URL=http://host.containers.internal:11434 \
    --name llamastack \
    llamastack/distribution-ollama:0.2.6
```

```shell
podman logs -f -n llamastack
```
4. Test the client

With the cli:

```shell
conda activate stack-client
llama-stack-client configure --endpoint http://localhost:8321 --api-key none
llama-stack-client models list
```

You should see the models table

With the play app:

```shell
conda activate stack-client
python llama-play/one_app.py
```

5. Build the container

```shell
podman build --tag fercoli/llama-container-test .
```

6. Optionally tag and push to quay.io

```shell
podman login quay.io
# username fercoli
podman login registry.redhat.io
# username fercoli@redhat.com
```

```shell
podman tag fercoli/llama-container-test quay.io/fercoli/llama-container-test
```

```shell
podman push quay.io/fercoli/llama-container-test
```

7. Run the container

```shell
podman run -it \
  -p 8000:8000 \
  --network=host \
  fercoli/llama-container-test
```

8. Http get

```shell
http http://0.0.0.0:8000/
```