from llama_stack_client import LlamaStackClient
from rich.pretty import pprint


def main():
    client = LlamaStackClient(base_url="http://localhost:8321")
    # List available models
    models = client.models.list()
    pprint(models)

    shields = client.shields.list()
    pprint(shields)

    toolgroups = client.toolgroups.list()
    pprint(toolgroups)

    # Select the first LLM
    llm = next(m for m in models if m.model_type == "llm")
    model_id = llm.identifier

    query_the_model(client, model_id, "Who won the 2022 Super Bowl?")
    query_the_model(client, model_id, "Who is the current US President?")


def query_the_model(client, model_id, user_query):
    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You're a helpful assistant."},
            {
                "role": "user",
                "content": user_query,
            },
        ],
        temperature=0.0,
    )
    response = completion.choices[0].message.content
    print(f"{user_query}: ", response)


if __name__ == "__main__":
    main()