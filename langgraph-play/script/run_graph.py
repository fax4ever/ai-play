from pathlib import Path

from langgraph_play.state import StateGraphBuilder


if __name__ == "__main__":
    builder = StateGraphBuilder()
    png_bytes = builder.graph.get_graph().draw_mermaid_png()
    
    output_path = Path("artifacts/graph.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png_bytes)
    print(f"Saved graph image to {output_path}")
        