import logging
from pathlib import Path
from langgraph_play.state import StateGraphBuilder


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    builder = StateGraphBuilder()
    graph = builder.graph.get_graph()
    mermaid_text = graph.draw_mermaid()
    png_bytes = graph.draw_mermaid_png()

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(mermaid_text)

    mmd_path = output_dir / "graph.mmd"
    mmd_path.write_text(mermaid_text, encoding="utf-8")
    print(f"Saved Mermaid source to {mmd_path}")

    png_path = output_dir / "graph.png"
    png_path.write_bytes(png_bytes)
    print(f"Saved graph image to {png_path}")
        