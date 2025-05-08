# Llama Play

Taken from https://llama-stack.readthedocs.io/

## Ollama

1. Install

See https://ollama.com/download

```
curl -fsSL https://ollama.com/install.sh | sh
```

2. Disable services

```
sudo systemctl status ollama.service
sudo systemctl stop ollama.service
sudo systemctl disable ollama.service
```

3. Start on demand

```
sudo systemctl start ollama.service
```

4. Run standalone Ollama

```
ollama run llama3.2:3b --keepalive 60m
```

## UV

1. Install

See https://docs.astral.sh/uv/

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. SEVER Run out of the BOX

```
INFERENCE_MODEL=llama3.2:3b uv run --with llama-stack llama stack build --template ollama --image-type venv --run
```

3. SERVER Create env && run it

```
uv venv --python 3.10
source .venv/bin/activate
pip install llama-stack
INFERENCE_MODEL=llama3.2:3b llama stack build --template ollama --image-type venv --run
```

4. CLIENT Create env && run it

```
uv venv client --python 3.10
source client/bin/activate
pip install llama-stack-client
```

## Conda (alternative)

5. SERVER with conda

```shell
yes | conda create -n llama python=3.10
conda activate llama
pip install llama-stack
INFERENCE_MODEL=llama3.2:3b llama stack build --template ollama --image-type conda  --image-name llama3-3b-conda --run
```

6. CLIENT with conda

```shell
yes | conda create -n stack-client python=3.10
conda activate stack-client
pip install llama-stack-client
```

## Llama client

```
llama-stack-client configure --endpoint http://localhost:8321 --api-key none
```

```
llama-stack-client models list
```

```
llama-stack-client inference chat-completion --message "Does Llama stack provide the RoBERTa model?"
```

## Try the basic app

With the **Llama client** profile activated:
```shell
python3 one_app.py
```

# Extra

## Conda get envs

```shell
conda info --envs
```