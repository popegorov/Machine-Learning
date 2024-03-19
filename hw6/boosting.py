from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        n = x.shape[0]
        idx = np.random.choice(n, size=int(n * self.subsample), replace=True)
        s = -(self.loss_derivative(y[idx], predictions[idx]))
        x_bootstrap = x[idx]
        model = self.base_model_class(**self.base_model_params)
        model.fit(x_bootstrap, s)
        cur_pred = model.predict(x)

        self.gammas.append(self.learning_rate * self.find_optimal_gamma(y, predictions, cur_pred))
        self.models.append(model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        counter = 0
        max_score = 0

        for _ in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.gammas[-1] * self.models[-1].predict(x_train)
            valid_predictions += self.gammas[-1] * self.models[-1].predict(x_valid)
            self.history['train'].append(self.loss_fn(y_train, train_predictions))
            self.history['valid'].append(self.loss_fn(y_train, train_predictions))

            if self.early_stopping_rounds is not None:
                probas = self.sigmoid(valid_predictions)
                cur_score = roc_auc_score(y_valid, probas)

                if cur_score > max_score:
                    max_score = cur_score
                else:
                    counter += 1
                    if counter == self.early_stopping_rounds:
                        break

        if self.plot:
            sns.lineplot(x=np.arange(len(self.history['train'])), y=self.history['train'], label='train')
            sns.lineplot(x=np.arange(len(self.history['valid'])), y=self.history['valid'], label='validation')

    def predict_proba(self, x):
        preds = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            preds += gamma * model.predict(x)

        probs_1 = self.sigmoid(preds)
        probs_0 = 1 - probs_1
        return np.array([probs_0, probs_1]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        sum_of_importances = np.zeros(self.models[0].feature_importances_.shape)

        for model in self.models:
            sum_of_importances += model.feature_importances_

        average_importances = sum_of_importances / len(self.models)

        return average_importances / average_importances.sum()




