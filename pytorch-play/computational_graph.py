"""
Copied and adapted from the colab:
https://colab.research.google.com/drive/17M5c-c2aByFXYa7PGLcLDKo9BPk30dL_?usp=classroom_web#scrollTo=9HFCJvEvMG1N
by Prof. Fabrizio Silvestri
"""
import torch
from torchviz import make_dot
from rich.pretty import pprint


def main():
    computational_dag_1()
    computational_dag_2()


def computational_dag_1():
    x = torch.tensor(2 * torch.pi / 3, requires_grad=True)
    pprint(x)
    y = torch.sin(x)
    pprint(y)
    y = y + 3 * torch.exp(-x)
    pprint(y)
    # Compute the gradient
    y.backward()
    # Print results
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"Gradient (dy/dx): {x.grad}")

    print(f"Gradient of y (non-leaf): {y.grad}")

    # Computational graph
    dot = make_dot(y, params={'x': x})
    dot.render("computation_graph", format="png", view=True)


def computational_dag_2():
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = (x * y)**2
    z.backward()

    gradient_x = x.grad
    gradient_y = y.grad
    print("Gradient of x: ", gradient_x)
    print("Gradient of y: ", gradient_y)

    print(f"Gradient of z (non-leaf): {z.grad}")

    # Computational graph
    dot = make_dot(z, params={'x': x})
    dot.render("computation_graph_x", format="png", view=True)
    dot = make_dot(z, params={'y': y})
    dot.render("computation_graph_y", format="png", view=True)


if __name__ == "__main__":
    main()
