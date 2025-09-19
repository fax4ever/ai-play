from llama_stack_client import LlamaStackClient
from opentelemetry import trace
import uuid
from interactive.auto_tracing import run as auto_tracing
from interactive.new_tracing import run1 as new_tracing

class InteractiveClient:
    def __init__(self):
        self._client = LlamaStackClient(base_url="http://localhost:8321")
        auto_tracing()

    def list_models(self):
        tracer = trace.get_tracer("Python LlamaStack application")

        with tracer.start_as_current_span(f"Python LlamaStack request - {uuid.uuid4()}"):
            return self._client.models.list()