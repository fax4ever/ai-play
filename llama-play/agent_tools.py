from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger
from rich.pretty import pprint


def main():
    # Copied and adapted from
    # https://llama-stack.readthedocs.io/en/latest/building_applications/tools.html

    client = LlamaStackClient(base_url="http://localhost:8321")
    all_tools = client.tools.list()
    pprint(all_tools)

    # List available models
    models = client.models.list()
    pprint(models)

    # Select the first LLM
    llm = next(m for m in models if m.model_type == "llm")
    model_id = llm.identifier

    agent = Agent(
        client,
        instructions="""
        You are a highly reliable, concise, and precise assistant.
        Always show the generated code, never generate your own code, and never anticipate results.
        """,
        model=model_id,
        tools=["builtin::code_interpreter"],
        max_infer_iters=5,
    )

    # Start a session
    session_id = agent.create_session("tool_session")

    # Send a query to the AI agent for code execution
    stream = agent.create_turn(
        messages=[{"role": "user", "content": "Run this code: print(3 ** 4 - 5 * 2)"}],
        session_id=session_id, stream=True
    )
    for event in AgentEventLogger().log(stream):
        event.print()

    print(3 ** 4 - 5 * 2)


if __name__ == "__main__":
    main()
