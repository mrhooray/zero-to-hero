import numpy as np
from ..nn import MLP
from ..train import train
from ..value import Value


class TestMLP:
    def test_mlp_backpropagation_training(self):
        np.random.seed(24)
        X = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
        y = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]

        mlp = MLP(2, [4, 1])

        losses = train(
            mlp,
            X,
            y,
            lambda pred, target: (pred - target) ** 2,
            epochs=100,
            learning_rate=0.05,
            verbose=False,
        )

        initial_loss = losses[0]
        final_loss = losses[-1]
        assert final_loss < initial_loss, (
            f"Loss should decrease: {initial_loss} -> {final_loss}"
        )

        gradients = [abs(p.grad) for p in mlp.parameters()]
        assert any(g > 1e-6 for g in gradients), "Some gradients should be non-zero"

        test_input = [Value(0), Value(1)]
        prediction = mlp(test_input)
        assert isinstance(prediction, Value), "MLP should return a Value"
        assert 0 < prediction.data < 1, "Prediction should be reasonable"
