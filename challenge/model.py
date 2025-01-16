import pandas as pd
import numpy as np

from typing import Tuple, Union, List
from joblib import dump, load
from datetime import datetime
from sklearn.linear_model import LogisticRegression


class DelayModel:
    _FEATURE_COLUMNS = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    @property
    def get_feature_cols(self):
        return self._FEATURE_COLUMNS

    def __init__(
        self
    ):
        self._model = LogisticRegression() # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data['Fecha-O'] = pd.to_datetime(data['Fecha-O'], format='%Y-%m-%d %H:%M:%S')
        data['Fecha-I'] = pd.to_datetime(data['Fecha-I'], format='%Y-%m-%d %H:%M:%S')

        data['min_diff'] = (data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60

        threshold_in_minutes = 15
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

        if target_column:
            target = pd.DataFrame(data[target_column])

        features = pd.concat(
            [
                pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
                pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'),
                pd.get_dummies(data['MES'], prefix = 'MES')
            ],
            axis = 1
        )

        features = features[self.get_feature_cols]

        if target_column:
            return (features, target)
        else:
            return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        n_y0 = len(target[target["delay"] == 0])
        n_y1 = len(target[target["delay"] == 1])

        class_weight = {1: n_y0/len(target), 0: n_y1/len(target)}
        self._model.class_weight = class_weight

        self._model.fit(features, target)

        dump(self._model, "flight_delay_logreg_model.joblib")


    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """

        self._model = load("flight_delay_logreg_model.joblib")

        prediction = self._model.predict(features)

        return prediction.tolist()
