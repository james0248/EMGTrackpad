import unittest
from io import StringIO

from rich.console import Console

from emg.inference.visualizer import InferenceState, TerminalVisualizer


class TerminalVisualizerTests(unittest.TestCase):
    def test_build_probability_bar_scales_with_probability(self):
        visualizer = TerminalVisualizer()
        half_bar = visualizer._build_probability_bar(0.5, width=10)
        self.assertIn("█████", half_bar)
        self.assertTrue(half_bar.endswith("░░░░░"))

    def test_controller_action_probabilities_are_rendered(self):
        visualizer = TerminalVisualizer(title="Test")
        state = InferenceState(
            action_names=["Move", "Scroll", "Left Click", "Right Click"],
            action_probs=[0.0, 0.5, 0.75, 1.0],
            buffer_ready=True,
            buffer_fill=1.0,
        )

        panel = visualizer.build_display(state)
        output = StringIO()
        console = Console(record=True, width=160, file=output)
        console.print(panel)
        rendered = console.export_text()

        for label in ["Move", "Scroll", "Left Click", "Right Click"]:
            self.assertIn(label, rendered)
        for value in ["0.0%", "50.0%", "75.0%", "100.0%"]:
            self.assertIn(value, rendered)
        self.assertIn("Action Probs", rendered)
        self.assertIn("█", rendered)


if __name__ == "__main__":
    unittest.main()
