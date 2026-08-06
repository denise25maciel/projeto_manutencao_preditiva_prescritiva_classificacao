import unittest

from avaliacao import avaliar
from prep import carregar, janelar, SINAIS


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

    def test_janelar_preserva_janela_bruta_e_features(self):
        df = carregar()
        janelas = janelar(df)

        self.assertIn("janela_bruta", janelas.columns)
        self.assertEqual(janelas.iloc[0]["janela_bruta"].shape[0], 30)
        self.assertEqual(janelas.iloc[0]["janela_bruta"].shape[1], len(SINAIS))
        self.assertEqual(len(janelas.iloc[0]["features"]), 90)

    def test_avaliar_retorna_classe_real_e_prevista(self):
        resultados = avaliar(n_estimators=40)

        self.assertTrue(resultados)
        self.assertIn("y_true", resultados[0]["folds"][0])
        self.assertIn("y_pred", resultados[0]["folds"][0])


if __name__ == "__main__":
    unittest.main()
