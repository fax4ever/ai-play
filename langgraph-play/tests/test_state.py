from langgraph_play.state import StateGraphBuilder


def test_state():
    builder = StateGraphBuilder()
    assert builder.graph is not None