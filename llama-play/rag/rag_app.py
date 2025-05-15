from llama_stack_client import LlamaStackClient
from rich.pretty import pprint


def log_db_info(client):
    vector_dbs = client.vector_dbs.list()
    for vector_db in vector_dbs:
        pprint(f"Vector DB: {vector_db.identifier}")

    providers = client.providers.list()
    for provider in providers:
        if provider.api == "vector_io":
            pprint(f"Vector DB Provider: {provider.provider_id}")


def main():
    client = LlamaStackClient(base_url="http://localhost:8321")
    providers_list = client.providers.list()
    pprint(providers_list)
    models = client.models.list()
    pprint(models)

    vector_db_id = "ragged-db"
    provider_id = "faiss"
    _ = client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_dimension=384,
        embedding_model="all-MiniLM-L6-v2",
        provider_id=provider_id
    )
    log_db_info(client)

    user_query = "What is the Grand Invention?"
    rag_response = client.tool_runtime.rag_tool.query(content=user_query, vector_db_ids=[vector_db_id])
    pprint(rag_response.content)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f""" 
            Answer the question based on the context provided.
            Context: {rag_response.content}
            Question: {user_query} 
            """,
        },
    ]

    completion = client.chat.completions.create(
        model='llama3.2:3b',
        messages=messages,
        temperature=0.1,
    )

    response = completion.choices[0].message.content
    print(f"{user_query}: ", response)

if __name__ == "__main__":
    main()
