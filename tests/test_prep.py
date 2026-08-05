import unittest

from prep import carregar


class PrepTests(unittest.TestCase):
    def test_carregar_dataframe(self):
        df = carregar()

        self.assertEqual(df.shape[0], 166796)
        self.assertEqual(df.shape[1], 26)
        self.assertEqual(df.isna().sum().sum(), 0)
        self.assertIn("created_at", df.columns)
        self.assertTrue(df["created_at"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
