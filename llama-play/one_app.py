import uuid

from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger
from llama_stack_client.types import Document
from rich.pretty import pprint


def main():
    # Copied and adapted from
    # https://llama-stack.readthedocs.io/en/latest/getting_started/detailed_tutorial.html

    client = LlamaStackClient(base_url="http://localhost:8321")
    # List available models
    models = client.models.list()
    pprint(models)

    # Select the first LLM
    llm = next(m for m in models if m.model_type == "llm")
    model_id = llm.identifier

    # Simple inference interaction
    response = client.inference.chat_completion(
        model_id=model_id,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Does Llama stack provide support also for encoder models?"},
        ],
    )
    print(response.completion_message.content)

    # Agent
    agent = Agent(client, model=model_id, instructions="You are a helpful assistant.")
    s_id = agent.create_session(session_name=f"s{uuid.uuid4().hex}")
    print("session ID: ", s_id)
    print("Streaming with print helper...")
    stream = agent.create_turn(
        messages=[{"role": "user", "content": "Who are you?"}], session_id=s_id, stream=True
    )
    for event in AgentEventLogger().log(stream):
        event.print()

    # RAG
    # Create a vector database instance
    embed_lm = next(m for m in client.models.list() if m.model_type == "embedding")
    embedding_model = embed_lm.identifier
    vector_db_id = f"v{uuid.uuid4().hex}"
    client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model,
    )

    # Create Documents
    urls = [
        "memory_optimizations.rst",
        "chat.rst",
        "llama3.rst",
        "qat_finetune.rst",
        "lora_finetune.rst",
    ]
    documents = [
        Document(
            document_id=f"num-{i}",
            content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
            mime_type="text/plain",
            metadata={},
        )
        for i, url in enumerate(urls)
    ]
    # Ingestion
    client.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=vector_db_id,
        chunk_size_in_tokens=512,
    )

    # Create the RAG agent
    rag_agent = Agent(
        client,
        model=model_id,
        instructions="You are a helpful assistant. Use the RAG tool to answer questions as needed.",
        tools=[
            {
                "name": "builtin::rag/knowledge_search",
                "args": {"vector_db_ids": [vector_db_id]},
            }
        ],
    )

    session_id = rag_agent.create_session(session_name=f"s{uuid.uuid4().hex}")

    turns = ["what is torchtune", "tell me about dora"]

    for t in turns:
        print("user>", t)
        stream = rag_agent.create_turn(
            messages=[{"role": "user", "content": t}], session_id=session_id, stream=True
        )
        for event in AgentEventLogger().log(stream):
            event.print()


if __name__ == "__main__":
    main()
