from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class Encoder:
    def __init__(self):
        self._label_encoder = LabelEncoder()
        self._one_hot_encoder = OneHotEncoder()

    def fit(self, x):
        x_labeled = self._label_encoder.fit_transform(x)
        self._one_hot_encoder.fit(x_labeled)

    def fit_transform(self, x):
        x_labeled = self._label_encoder.fit_transform(x).reshape(-1, 1)
        x_one_hot = self._one_hot_encoder.fit_transform(x_labeled).toarray()

        return x_one_hot

    def transform(self, x):
        return self._one_hot_encoder.transform(self._label_encoder.transform(x))
