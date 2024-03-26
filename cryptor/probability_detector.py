import os
import pickle
from typing import Any
from cryptor.detector import Detector, DetectorOutput
import numpy as np
from common.llama_service import LlamaService
from scipy.stats import beta, entropy, wasserstein_distance, zscore, gaussian_kde, FitError
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class ProbabilityDetector(Detector):
    MAX_LEN = 64
    N_BOOTSTRAPS = 7

    def __init__(self, n_threads) -> None:
        self._llama_service = LlamaService(
            context_size=512, n_threads=n_threads)

        self.alpha, self.beta = 0.35, 0.36

        with open(os.path.join(os.path.dirname(__file__), "classifier.pkl"), "rb") as f:
            self._classifier = pickle.load(f)

    def detect(self, news_feed: list[str]) -> DetectorOutput:
        results = []
        all_perplexities = []
        for feed in news_feed:
            overall_score, perplexities = self._llama_service.get_perplexity(
                feed)

            all_perplexities.extend(perplexities)
            X = [self.compute_metrics(np.array(perplexities))]
            result = self._classifier.predict(X)[0]
            results.append(bool(result))

        X = self.compute_metrics(np.asarray(all_perplexities))
        overall_result = self._classifier.predict([X])[0]

        found_something = any(results) or bool(overall_result)
        return DetectorOutput(
            contains_secret=found_something,
            reconstructed_secret="",
            statistics=None,
        )

    def _bootstrap_alpha_beta(self, probs: np.ndarray) -> tuple[float, float]:
        here_alpha, here_beta = 0, 0
        divi = self.N_BOOTSTRAPS

        for _ in range(self.N_BOOTSTRAPS):
            drawn_probs = np.random.choice(
                probs, probs.shape[0] + 10, replace=True)
            try:
                alpha, beta_, _, _ = beta.fit(drawn_probs, floc=0, fscale=1)
            except FitError:
                divi -= 1
                continue

            here_alpha += alpha
            here_beta += beta_

        if divi <= 0:
            return 0.5, 0.5

        return here_alpha / divi, here_beta / divi

    def compute_metrics(self, probs: np.ndarray) -> np.ndarray:
        probs = probs[probs != -100]
        here_alpha, here_beta = self._bootstrap_alpha_beta(probs)

        x = np.linspace(1e-5, 0.99999, 1000)
        y = beta.pdf(x, here_alpha, here_beta, loc=0, scale=1)
        y_true = beta.pdf(x, self.alpha, self.beta, loc=0, scale=1)

        kl_div = entropy(y_true, y)
        true_samples = beta.rvs(self.alpha, self.beta,
                                size=len(probs), loc=0, scale=1)
        wasser_dist = wasserstein_distance(true_samples, probs)
        z_score = zscore(probs).mean() - zscore(true_samples).mean()

        try:
            kde_estimate = gaussian_kde(probs)
            kl_div_kde = entropy(y_true, kde_estimate(x))
        except np.linalg.LinAlgError:
            kl_div_kde = -1

        combined = np.asarray([kl_div, wasser_dist, z_score, kl_div_kde])
        combined[combined == np.inf] = combined[combined != np.inf].max() + 10
        combined[np.isnan(combined)] = -1

        return combined

    def _batch_compute_metrics(self, probs: np.ndarray) -> np.ndarray:
        return np.asarray(list(map(self.compute_metrics, probs)))

    def train(self, X: list[np.ndarray], y: np.ndarray):
        max_len = max([len(i) for i in X])

        for i in range(len(X)):
            X[i] = np.concatenate([X[i], np.ones(max_len - len(X[i])) * -100])

        X = np.asarray(X)
        X = self._batch_compute_metrics(X)

        svc_pipeline = Pipeline([
            ("b", StandardScaler()),
            ("svc", SVC())]
        )
        logistic_regression_pipeline = Pipeline([
            ("b", StandardScaler()),
            ("lr", LogisticRegression())]
        )
        gradient_boosting_pipeline = Pipeline([
            ("b", StandardScaler()),
            ("xgb", GradientBoostingClassifier())]
        )
        lda_pipeline = Pipeline([
            ("b", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis())]
        )

        estimatos = {
            "SVC": svc_pipeline,
            "LogisticRegression": logistic_regression_pipeline,
            "GradientBoosting": gradient_boosting_pipeline,
            "LDA": lda_pipeline,
        }

        result_data = []
        from tqdm import tqdm
        for clf_name, clf in tqdm(estimatos.items()):
            result = cross_validate(clf, X, y, cv=4, n_jobs=4, scoring="f1")
            result_data.append(
                (result["test_score"].mean(), clf_name))

        best_pipeline_name = max(result_data, key=lambda x: x[0])[1]
        best_pipeline = estimatos[best_pipeline_name].fit(X, y)
        print(
            f"Best pipeline: {best_pipeline_name} with f1 score of {max(result_data, key=lambda x: x[0])[0]}")

        with open("cryptor/classifier.pkl", "wb") as f:
            pickle.dump(best_pipeline, f)
