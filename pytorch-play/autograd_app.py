"""
Copied and adapted from the colab:
https://colab.research.google.com/drive/17M5c-c2aByFXYa7PGLcLDKo9BPk30dL_?usp=classroom_web#scrollTo=9HFCJvEvMG1N
by Prof. Fabrizio Silvestri
"""

import torch


def f(x):
    return x**2 + 2*x + 1  # differentiable function


def autograd_example():
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


def inline_operations():
    x = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    y  = (x * 2).sum()
    print('gradient is maintained:', y.requires_grad)
    z = 3*y
    z.backward()
    print(x.grad)
    print(y.grad)


def main():
    autograd_example()
    grad_no_grad()
    inline_operations()


if __name__ == "__main__":
    main()
