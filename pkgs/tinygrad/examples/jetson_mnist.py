#!/usr/bin/env python3
"""
MNIST Training Example for NVIDIA Jetson using TinyGrad

This example demonstrates how to train a simple CNN model on MNIST
using TinyGrad with CUDA acceleration on Jetson devices.

Requirements:
  pip install -e /path/to/vendor/tinygrad
  pip install numpy pillow

Usage:
  # Use CUDA on Jetson
  CUDA=1 python jetson_mnist.py

  # Use CPU (for testing)
  python jetson_mnist.py

  # Fashion MNIST variant
  FASHION=1 CUDA=1 python jetson_mnist.py
"""

import math
import os
from typing import Callable

# Set the device before importing tinygrad
if os.getenv("CUDA", "0") == "1":
    os.environ["DEVICE"] = "CUDA"
    print("Using CUDA device (Jetson GPU)")
else:
    os.environ["DEVICE"] = "CPU"
    print("Using CPU device")

from tinygrad import Tensor, TinyJit, nn, GlobalCounters, Device
from tinygrad.helpers import getenv, colored, trange
from tinygrad.nn.datasets import mnist


class JetsonMNISTModel:
    """
    A simple but effective CNN model for MNIST classification.
    Optimized for memory efficiency on Jetson devices.
    """
    def __init__(self):
        self.layers: list[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(1, 32, 5), Tensor.relu,
            nn.Conv2d(32, 32, 5), Tensor.relu,
            nn.BatchNorm(32), Tensor.max_pool2d,
            nn.Conv2d(32, 64, 3), Tensor.relu,
            nn.Conv2d(64, 64, 3), Tensor.relu,
            nn.BatchNorm(64), Tensor.max_pool2d,
            lambda x: x.flatten(1), nn.Linear(576, 10)
        ]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


def main():
    print(f"TinyGrad Device: {Device.DEFAULT}")
    print("=" * 50)
    print("Loading MNIST dataset...")

    # Load dataset
    fashion = getenv("FASHION", 0)
    X_train, Y_train, X_test, Y_test = mnist(fashion=fashion)

    dataset_name = "Fashion MNIST" if fashion else "MNIST"
    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print("=" * 50)

    # Create model
    model = JetsonMNISTModel()

    # Select optimizer based on environment
    if getenv("MUON"):
        opt = nn.optim.Muon(nn.state.get_parameters(model))
        print("Optimizer: Muon")
    elif getenv("SGD"):
        opt = nn.optim.SGD(nn.state.get_parameters(model))
        print("Optimizer: SGD")
    else:
        opt = nn.optim.Adam(nn.state.get_parameters(model))
        print("Optimizer: Adam")

    batch_size = getenv("BS", 512)
    steps = getenv("STEPS", 70)
    print(f"Batch size: {batch_size}")
    print(f"Training steps: {steps}")
    print("=" * 50)

    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        opt.zero_grad()
        samples = Tensor.randint(batch_size, high=X_train.shape[0])
        loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
        return loss.realize(*opt.schedule_step())

    @TinyJit
    def get_test_acc() -> Tensor:
        return (model(X_test).argmax(axis=1) == Y_test).mean() * 100

    # Training loop
    print("Starting training...")
    test_acc = math.nan
    for i in (t := trange(steps)):
        GlobalCounters.reset()
        loss = train_step()
        if i % 10 == 9:
            test_acc = get_test_acc().item()
        t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

    # Final evaluation
    final_acc = get_test_acc().item()
    print("=" * 50)
    print(f"Training complete!")
    print(f"Final test accuracy: {final_acc:.2f}%")

    # Verify target accuracy
    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if final_acc >= target and final_acc != 100.0:
            print(colored(f"SUCCESS: {final_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"FAILED: {final_acc=} < {target}", "red"))


if __name__ == "__main__":
    main()
