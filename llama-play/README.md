# Llama Play

Copied, adapted from https://llama-stack.readthedocs.io/ and extend it...

1. Activate the Ollama service

```shell
sudo systemctl start ollama.service
```

2. Check if the service is active

```shell
sudo systemctl status ollama.service
```

3. Run Llama stack with a Container

Prepare the environment:

```shell
export INFERENCE_MODEL="llama3.2:3b"
export LLAMA_STACK_PORT=8321
mkdir -p ~/code/lab/.llama
```

Run it:

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