"""
Copied and adapted from the colab:
https://colab.research.google.com/drive/17M5c-c2aByFXYa7PGLcLDKo9BPk30dL_?usp=classroom_web#scrollTo=9HFCJvEvMG1N
by Prof. Fabrizio Silvestri
"""

import torch
from rich.pretty import pprint


BATCH_SIZE = 16
DIM_IN = 100
HIDDEN_SIZE = 100
DIM_OUT = 10


class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.layer1 = torch.nn.Linear(DIM_IN, HIDDEN_SIZE)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(HIDDEN_SIZE, DIM_OUT)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def training_step(some_input, ideal_output):
    model = TinyModel()
    pprint(model)
    print("model.layer1.weight.requires_grad: ", model.layer1.weight.requires_grad)
    print("model.layer2.weight.requires_grad: ", model.layer2.weight.requires_grad)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    prediction = model(some_input)
    # MSE loss:
    loss = (ideal_output - prediction).pow(2).sum()
    print("loss before the backward:", loss)
    print("grad before the backward:", model.layer2.weight.grad)
    print("weights before the backward:", model.layer2.weight[0][0:10])
    loss.backward()
    print("grad after the backward:", model.layer2.weight.grad[0][0:10])
    optimizer.step()
    print("weights after the optimizer step:", model.layer2.weight[0][0:10])
    prediction = model(some_input)
    # MSE loss:
    loss = (ideal_output - prediction).pow(2).sum()
    print("loss using the updated model:", loss)


def grad(model):
    current_grad = model.layer2.weight.grad
    if current_grad is None:
        return None
    return current_grad[0][0:10]


def gradient_accumulation(some_input, ideal_output):
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for i in range(0, 5):
        prediction = model(some_input)
        loss = (ideal_output - prediction).pow(2).sum()
        print("loss on training", i+1, "/5:", loss)
        loss.backward()
        optimizer.step()
        print("grad on training", i+1, "/5:", grad(model))
        # the gradients on the learning weights will accumulate:
        # optimizer.zero_grad()


def training_loop(some_input, ideal_output):
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for i in range(0, 5):
        prediction = model(some_input)
        loss = (ideal_output - prediction).pow(2).sum()
        print("loss on training", i+1, "/5:", loss)
        loss.backward()
        optimizer.step()
        print("grad on training", i + 1, "/5:", grad(model))
        optimizer.zero_grad()


def main():
    some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
    ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)
    training_step(some_input, ideal_output)
    gradient_accumulation(some_input, ideal_output)
    training_loop(some_input, ideal_output)


if __name__ == "__main__":
    main()
