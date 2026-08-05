import unittest

from interface_app import build_summary
from prep import carregar


class InterfaceTests(unittest.TestCase):
    def test_build_summary(self):
        df = carregar()
        summary = build_summary(df)

        self.assertEqual(summary["linhas"], 166796)
        self.assertEqual(summary["colunas"], 26)
        self.assertEqual(summary["valores_ausentes"], 0)
        self.assertTrue(summary["ordenado"])
        self.assertIn("fault", summary["top_faults"])


if __name__ == "__main__":
    unittest.main()
