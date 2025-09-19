from interactive.client import InteractiveClient


def main():
    print("Hello from llama-stack-app!")
    interactive = InteractiveClient()
    print(interactive.list_models())


if __name__ == "__main__":
    main()
