"""
Copied and adapted from the colab:
https://colab.research.google.com/drive/17M5c-c2aByFXYa7PGLcLDKo9BPk30dL_?usp=classroom_web#scrollTo=9HFCJvEvMG1N
by Prof. Fabrizio Silvestri
"""

import torch


def f(x):
    return x**2 + 2*x + 1 # differentiable function


def autogradient_example():
    # enables automatic differentiation
    # used for parameters:
    x = torch.tensor(2.0, requires_grad=True)
    print(x)
    # this is a static value
    # used for data
    x_ = torch.tensor(2.0)
    print(x_)
    # we search for x : minimize y
    y = f(x)
    y.backward()
    # the gradient is computed by the autograd
    gradient = x.grad
    # We access the gradient of 'y' with respect to 'x', which represents the derivative of 'f(x)' with respect to 'x'.
    print("Function value (f(x)): ", y.item())
    print("Gradient (df/dx) at x=2: ", gradient.item())


def grad_no_grad():
    x = torch.ones(5)  # input tensor
    y = torch.zeros(3)  # expected output
    w = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)

    z = torch.matmul(x, w)+b
    print(z.requires_grad)

    with torch.no_grad():
        z = torch.matmul(x, w)+b
    print(z.requires_grad)

    z = torch.matmul(x, w)+b
    z_det = z.detach()
    print(z_det.requires_grad)


def main():
    autogradient_example()
    grad_no_grad()


if __name__ == "__main__":
    main()