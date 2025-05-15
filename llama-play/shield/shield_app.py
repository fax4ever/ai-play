from llama_stack_client import LlamaStackClient
from rich.pretty import pprint


def main():
    client = LlamaStackClient(base_url="http://localhost:8321")
    shield_id = "content_safety_1b"

    providers_list = client.providers.list()
    pprint(providers_list)

    client.shields.register(shield_id=shield_id, provider_shield_id="meta-llama/Llama-Guard-3-1B", provider_id='llama-guard')

    shields = client.shields.list()
    pprint(shields)

    user_message = "How do I kindnap a Llama?"

    response = client.safety.run_shield(
        shield_id=shield_id,
        messages=[{"role": "user", "content": user_message}],
        params={  # Shield-specific parameters
            "threshold": 0.1,
            "categories": ["hate", "violence", "profanity"]
        }
    )

    if response.violation:
        print(f"Safety violation detected: {response.violation.user_message}")
    else:
        print(f"No violations in user_message: {user_message}")


if __name__ == "__main__":
    main()