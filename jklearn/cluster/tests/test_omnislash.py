from unittest import TestCase

import numpy as np
import pandas as pd
import requests

from jklearn.cluster import Omnislash
from jklearn.cluster.omnislash import percentile
from jklearn.cluster.omnislash import mean_and_var
from jklearn.cluster.omnislash import evaluate_score


class TestOmnislash(TestCase):

    @classmethod
    def setUpClass(cls):
        data = []
        r = requests.get("http://cs.joensuu.fi/sipu/datasets/s2.txt")
        for line in r.text.split("\n"):
            if line.strip() == "":
                continue
            x, y = line.strip().split()
            data.append({"x": float(x), "y": float(y)})
        df = pd.DataFrame(data)

        data = []
        args = [(5, 4, 10000), (-5, 4, 10000)]
        for loc, scale, size in args:
            data += list(np.random.normal(loc=loc, scale=scale, size=size))

        cls.cluster_data = df.as_matrix()
        cls.double_bump_data = np.sort(np.array(data))
        cls.lin_space_data = np.linspace(1, 100, 100)

    def test_percentile(self):
        for i in range(1, 99):
            self.assertAlmostEqual(percentile(self.lin_space_data, i/100),
                                   np.percentile(self.lin_space_data, i))
        with self.assertRaises(IndexError):
            percentile(self.lin_space_data, 10)

    def test_mean_and_var(self):
        mean, var = mean_and_var(self.double_bump_data)
        self.assertAlmostEqual(mean, np.mean(self.double_bump_data))
        self.assertAlmostEqual(var, np.var(self.double_bump_data))

    def test_evaluate_score(self):
        min_cut = None
        min_score = None
        for cut in np.arange(min(self.double_bump_data),
                             max(self.double_bump_data)+1, 0.1):
            score = evaluate_score(cut, self.double_bump_data)
            if (cut == min(self.double_bump_data) or
                cut == max(self.double_bump_data)):
                self.assertNotEqual(score, score)
            else:
                if min_score is None or score < min_score:
                    min_score = score
                    min_cut = cut
            print(cut, score)
        self.assertLess(min_cut, 1)
        self.assertGreater(min_cut, -1)

    def test_omnislash(self):
        omni = Omnislash(50)
        labels = omni.fit_predict(self.cluster_data)
        self.assertEqual(len(set(labels)), 17)

        omni = Omnislash(50, n_components_max=1000)
        omni.fit_predict(self.cluster_data)
        self.assertEqual(omni.pca.n_components, 2)
