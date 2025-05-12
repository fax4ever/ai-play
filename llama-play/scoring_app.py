from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger
from rich.pretty import pprint


def main():
    # Copied and adapted from
    # https://llama-stack.readthedocs.io/en/latest/building_applications/evals.html#application-evaluation

    client = LlamaStackClient(base_url="http://localhost:8321")

    # Select the first LLM
    models = client.models.list()
    llm = next(m for m in models if m.model_type == "llm")
    model_id = llm.identifier

    agent = Agent(
        client,
        model=model_id,
        instructions="You are a helpful assistant. Use search tool to answer the questions. ",
        tools=["builtin::websearch"],
    )
    user_prompts = [
        "Which teams played in the NBA Western Conference Finals of 2024. Search the web for the answer.",
        "In which episode and season of South Park does Bill Cosby (BSM-471) first appear? Give me the number and title. Search the web for the answer.",
        "What is the British-American kickboxer Andrew Tate's kickboxing name? Search the web for the answer.",
    ]

    session_id = agent.create_session("test-session")

    for prompt in user_prompts:
        response = agent.create_turn(
            messages=[{"role": "user", "content": prompt,}],
            session_id=session_id,
        )

        for log in AgentEventLogger().log(response):
            log.print()

    session_response = client.agents.session.retrieve(
        session_id=session_id,
        agent_id=agent.agent_id,
    )

    pprint(session_response)

    eval_rows = []

    expected_answers = [
        "Dallas Mavericks and the Minnesota Timberwolves",
        "Season 4, Episode 12",
        "King Cobra",
    ]

    for i, turn in enumerate(session_response.turns):
        eval_rows.append(
            {
                "input_query": turn.input_messages[0].content,
                "generated_answer": turn.output_message.content,
                "expected_answer": expected_answers[i],
            }
        )

    pprint(eval_rows)

    scoring_params = {
        "basic::subset_of": None,
    }
    scoring_response = client.scoring.score(
        input_rows=eval_rows, scoring_functions=scoring_params
    )
    pprint(scoring_response)

if __name__ == "__main__":
    main()