"""Small feedforward neural network implemented directly with numpy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class NeuralNetwork:
    """One-hidden-layer policy network with tanh and softmax activations."""

    input_size: int
    hidden_size: int
    output_size: int
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def random(
        cls,
        input_size: int,
        hidden_size: int,
        output_size: int,
        rng: np.random.Generator,
    ) -> "NeuralNetwork":
        """Create a network with Xavier-style random weights."""

        w1 = rng.normal(0.0, np.sqrt(1.0 / input_size), size=(input_size, hidden_size))
        b1 = np.zeros(hidden_size, dtype=float)
        w2 = rng.normal(0.0, np.sqrt(1.0 / hidden_size), size=(hidden_size, output_size))
        b2 = np.zeros(output_size, dtype=float)
        return cls(input_size, hidden_size, output_size, w1, b1, w2, b2)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Return action probabilities for a single observation vector."""

        hidden = np.tanh(inputs @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        return self._softmax(logits)

    def act(self, inputs: np.ndarray, rng: np.random.Generator) -> int:
        """Sample an action from the policy distribution."""

        probabilities = self.forward(inputs)
        return int(rng.choice(self.output_size, p=probabilities))

    def policy(
        self,
        inputs: np.ndarray,
        action_count: int,
        rng: np.random.Generator,
        action_bias: np.ndarray | None = None,
    ) -> tuple[int, float]:
        """Sample an action and emit a communication signal.

        The first `action_count` outputs form the action distribution. If the
        network has one extra output, that final logit is converted with tanh
        into a broadcast signal in [-1, 1].
        """

        hidden = np.tanh(inputs @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        action_logits = logits[:action_count]
        if action_bias is not None:
            action_logits = action_logits + action_bias
        action_probabilities = self._softmax(action_logits)
        action = int(rng.choice(action_count, p=action_probabilities))
        signal = float(np.tanh(logits[action_count])) if self.output_size > action_count else 0.0
        return action, signal

    def clone(self) -> "NeuralNetwork":
        """Return an independent copy of the network."""

        return NeuralNetwork(
            self.input_size,
            self.hidden_size,
            self.output_size,
            self.w1.copy(),
            self.b1.copy(),
            self.w2.copy(),
            self.b2.copy(),
        )

    def mutate(
        self,
        rng: np.random.Generator,
        mutation_rate: float,
        mutation_strength: float,
    ) -> None:
        """Apply Gaussian parameter noise to a random subset of weights."""

        for parameter in self.parameters():
            mask = rng.random(parameter.shape) < mutation_rate
            noise = rng.normal(0.0, mutation_strength, size=parameter.shape)
            parameter += mask * noise

    def parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return all trainable arrays."""

        return self.w1, self.b1, self.w2, self.b2

    def flattened_parameters(self) -> np.ndarray:
        """Return all parameters as one vector for diversity metrics."""

        return np.concatenate([parameter.ravel() for parameter in self.parameters()])

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values)
