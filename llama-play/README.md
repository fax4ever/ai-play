# Llama Play

Copied, adapted from https://llama-stack.readthedocs.io/ and extend it...

0. Configure Ollama to accept any binding address (such as `host.docker.internal`)

By default, I think that the binding host is `localhost:11434`.
If you run Ollama as a service, add `Environment="OLLAMA_HOST=0.0.0.0"` to the `/etc/systemd/system/ollama.service`.

1. Run Ollama service

```shell
sudo systemctl start ollama.service
```

```shell
sudo systemctl status ollama.service
```

2. Alternatively, manually run the `ollama serve` command:

```shell
OLLAMA_HOST=0.0.0.0 ollama serve
```

3. Run Llama stack with a Container

Prepare the environment:

```shell
export INFERENCE_MODEL="llama3.2:3b"
export LLAMA_STACK_PORT=8321
mkdir -p ~/code/lab/.llama
```

With Docker, the command is taken from the Llama documentation.

```shell
docker run -it \
  --pull always \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/code/lab/.llama:/root/.llama \
  --network=host \
  llamastack/distribution-ollama \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://localhost:11434
```  

> **_NOTE:_** the `--network=host` allows to make `localhost` reachable from the guest container.

> **IMPORTANT_** with Podman we need to add `:z` to the mounting point of the `.llama` director

```shell
podman run -it \
  --pull always \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/code/lab/.llama:/root/.llama:z \
  --network=host \
  llamastack/distribution-ollama \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://localhost:11434
```

> **_NOTE:_** to use `--env OLLAMA_URL=http://host.docker.internal:11434` instead of `--network=host` and `localhost`, 
but the Ollama service must have `0.0.0.0` as binding address and not `localhost` as biding address.

If Ollama service has `0.0.0.0` as binding address we can run:

```shell
podman run -it \
  --pull always \
  -p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT \
  -v ~/.llama:/root/.llama:z \
  llamastack/distribution-ollama \
  --port $LLAMA_STACK_PORT \
  --env INFERENCE_MODEL=$INFERENCE_MODEL \
  --env OLLAMA_URL=http://host.docker.internal:11434
```

5. Run the client app

```shell
conda create -n llama-stack-client python=3.10
conda activate llama-stack-client
pip install llama-stack-client
```

Go to the root of this project `ai-play`

Run the client app:

```shell
python3 llama-play/one_app.py
```

# Extra

## Ollama

1. Install Ollama

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

See https://ollama.com/download

Also add the env entry:

```shell
Environment="OLLAMA_HOST=0.0.0.0"
```

to the service file `/etc/systemd/system/ollama.service`.

2. Disable services

```shell
sudo systemctl status ollama.service
sudo systemctl stop ollama.service
sudo systemctl disable ollama.service
```

3. Start on demand

```shell
sudo systemctl start ollama.service
```

4. Service process log

```shell
journalctl -u ollama
```

5. Pull explicitly a new model with Ollama

**Optional**: This is usually done automatically when a model is required.

```shell
ollama pull llama3.2:3b
```

6. Run standalone Ollama

this basically without the Llama stack

```shell
ollama run llama3.2:3b --keepalive 60m
```

## UV

1. Install

See https://docs.astral.sh/uv/

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. SEVER Run out of the BOX

```shell
INFERENCE_MODEL=llama3.2:3b uv run --with llama-stack llama stack build --template ollama --image-type venv --run
```

3. SERVER Create env && run it

```shell
uv venv --python 3.10
source .venv/bin/activate
pip install llama-stack
INFERENCE_MODEL=llama3.2:3b llama stack build --template ollama --image-type venv --run
```

4. CLIENT Create env && run it

```shell
uv venv client --python 3.10
source client/bin/activate
pip install llama-stack-client
```

## Conda

1. SERVER with conda

```shell
yes | conda create -n llama python=3.10
conda activate llama
INFERENCE_MODEL=llama3.2:3b llama stack build --template ollama --image-type conda --run
pip install e2b-code-interpreter # for [agent_tools.py](agent_tools.py)
```

2. CLIENT with conda

```shell
yes | conda create -n stack-client python=3.10
conda activate stack-client
pip install llama-stack-client
```

3. Conda get envs

```shell
conda info --envs
```

4. Conda delete evn

```shell
conda remove -n llama-play --all
```