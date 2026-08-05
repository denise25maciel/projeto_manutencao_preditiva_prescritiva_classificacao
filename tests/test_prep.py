import unittest

from prep import carregar


class PrepTests(unittest.TestCase):
    def test_carregar_dataframe(self):
        df = carregar()

        self.assertEqual(df.shape[0], 166796)
        self.assertEqual(df.shape[1], 27)
        self.assertEqual(df.isna().sum().sum(), 0)
        self.assertIn("created_at", df.columns)
        self.assertIn("segment_id", df.columns)
        self.assertTrue(df["created_at"].is_monotonic_increasing)

    def test_segmentacao(self):
        df = carregar()

        dt = df["created_at"].diff().dt.total_seconds()
        self.assertAlmostEqual(dt.median(), 2.0, delta=0.01)
        self.assertEqual((dt > 3600).sum(), 56)
        self.assertEqual(df["segment_id"].nunique(), 240)


if __name__ == "__main__":
    unittest.main()
