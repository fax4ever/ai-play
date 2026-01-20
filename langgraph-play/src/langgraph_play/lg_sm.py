#!/usr/bin/env python3
"""
Standalone copy of StateMachine for testing.
Stripped of external dependencies.
"""
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict
import re
import yaml
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


def create_agent_state_class(state_schema: dict[str, Any]) -> type[dict[str, Any]]:
    """Create a dynamic AgentState TypedDict class based on YAML configuration."""
    fields = {}
    fields["messages"] = Annotated[List[BaseMessage], add_messages]
    fields["current_state"] = str

    if not state_schema:
        state_schema = {}

    business_fields = state_schema.get("business_fields", {})
    for field_name, field_config in business_fields.items():
        field_type = field_config.get("type", "string")
        if field_type == "string":
            fields[field_name] = Optional[str]
        elif field_type == "list":
            fields[field_name] = List[Dict[str, Any]]
        elif field_type == "dict":
            fields[field_name] = Optional[Dict[str, Any]]
        elif field_type == "boolean":
            fields[field_name] = Optional[bool]
        else:
            fields[field_name] = Optional[str]

    fields["_last_processed_human_count"] = Optional[int]
    fields["_consumed_this_invoke"] = Optional[bool]
    fields["_last_waiting_node"] = Optional[str]

    return TypedDict("AgentState", fields)


class StateMachine:
    """Configurable state machine engine for conversation flows."""

    def __init__(self):
        config_path = Path(__file__).parent / "config" / "config.yaml"
        self.config_path = Path(config_path)
        self.config = self._load_config()
        state_schema = self.config.get("state_schema", {})
        self.AgentState = create_agent_state_class(state_schema)

    def _load_config(self) -> dict[str, Any]:
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
            return config if isinstance(config, dict) else {}

    def is_terminal_state(self, state_name: str) -> bool:
        settings = self.config.get("settings", {})
        terminal_state = settings.get("terminal_state", "end")
        return state_name == str(terminal_state)

    def is_waiting_state(self, state_name: str) -> bool:
        states = self.config.get("states", {})
        state_config = states.get(state_name, {})
        return state_config.get("type") == "waiting"

    def create_initial_state(self) -> dict[str, Any]:
        settings = self.config.get("settings", {})
        initial_state_name = settings.get("initial_state", "start")
        state_schema = self.config.get("state_schema", {})

        state: Dict[str, Any] = {
            "messages": [],
            "current_state": initial_state_name,
        }

        business_fields = state_schema.get("business_fields", {})
        for field_name, field_config in business_fields.items():
            default_value = field_config.get("default")
            if default_value == "null" or default_value is None:
                state[field_name] = [] if field_config.get("type") == "list" else None
            elif default_value == "false":
                state[field_name] = False
            elif default_value == "true":
                state[field_name] = True
            else:
                state[field_name] = default_value

        return state

    def _format_text(
        self,
        text: str,
        state_data: dict[str, Any],
        authoritative_user_id: str | None = None,
    ) -> str:
        format_data = dict(state_data)
        if authoritative_user_id:
            format_data["authoritative_user_id"] = authoritative_user_id

        # Extract last_user_message
        messages = state_data.get("messages", [])
        last_user_message = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and type(msg).__name__ == "HumanMessage":
                last_user_message = msg.content
                break
        format_data["last_user_message"] = last_user_message

        # Replace placeholders
        text = text.replace("{{", "\x00OPEN\x00").replace("}}", "\x00CLOSE\x00")
        
        def replacer(match):
            field_path = match.group(1)
            try:
                value = format_data
                for part in field_path.split("."):
                    value = value[part] if isinstance(value, dict) else getattr(value, part)
                return str(value)
            except (KeyError, AttributeError, TypeError):
                return match.group(0)

        text = re.sub(r"\{([^}]+)\}", replacer, text)
        return text.replace("\x00OPEN\x00", "{").replace("\x00CLOSE\x00", "}")
