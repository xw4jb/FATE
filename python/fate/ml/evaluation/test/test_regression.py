import unittest
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fate.ml.evaluation.metrics.regression import RMSE, MAE, R2Score, MSE


class TestMetrics(unittest.TestCase):

    def setUp(self):
        # Create example data for each test using np.random
        np.random.seed(123)
        self.labels = np.random.normal(size=100)
        self.pred_scores = np.random.normal(size=100)

    def test_rmse(self):
        rmse_metric = RMSE()
        rmse = rmse_metric(self.labels, self.pred_scores)
        print(rmse)
        self.assertAlmostEqual(rmse, np.sqrt(mean_squared_error(self.labels, self.pred_scores)), places=7)

    def test_mae(self):
        mae_metric = MAE()
        mae = mae_metric(self.labels, self.pred_scores)
        print(mae)
        self.assertAlmostEqual(mae, mean_absolute_error(self.labels, self.pred_scores), places=7)

    def test_r2score(self):
        r2_metric = R2Score()
        r2 = r2_metric(self.labels, self.pred_scores)
        print(r2)
        self.assertAlmostEqual(r2, r2_score(self.labels, self.pred_scores), places=7)

    def test_mse(self):
        mse_metric = MSE()
        mse = mse_metric(self.labels, self.pred_scores)
        print(mse)
        self.assertAlmostEqual(mse, mean_squared_error(self.labels, self.pred_scores), places=7)


if __name__ == '__main__':
    unittest.main()
