import unittest

import numpy as np
import torch

from emg.inference.output_interface import ControllerModelInterface


class ControllerModelInterfaceTests(unittest.TestCase):
    def test_get_action_probs_returns_sigmoid_logits(self):
        interface = ControllerModelInterface(
            dxdy_mean=np.zeros(4, dtype=np.float64),
            dxdy_std=np.ones(4, dtype=np.float64),
        )
        logits = np.array([0.0, np.log(3.0), -np.log(3.0), 2.0], dtype=np.float32)
        model_output = {
            "dxdy": torch.zeros((1, 4), dtype=torch.float32),
            "actions": torch.from_numpy(logits).unsqueeze(0),
        }

        probs = interface.get_action_probs(model_output)
        expected = 1.0 / (1.0 + np.exp(-logits))
        self.assertTrue(np.allclose(probs, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
