def debug(x):
    print("===== [DEBUG] =====")
    if isinstance(x, list):
        for i in x:
            print(i)
    else:
        print(x)


from sklearn.preprocessing import StandardScaler

import joblib
import os


class Scaler:
    def __init__(self, keys, create=False, path="./data/scaler/"):
        self.scaler = {}
        # self._scale = scale
        self._keys = keys
        self._path = path
        self._creatScaler(create)

    def __call__(self):
        return self.scaler

    @property
    def keys(self):
        return self._keys

    def _creatScaler(self, create):
        if not os.path.exists(self._path):
            os.makedirs(self._path)

        dir_list = os.listdir(self._path)

        for k in self._keys:
            self.scaler[k] = None
            for dir in dir_list:
                if k in dir:
                    print(f"[Load-{k}] : ", self._path + dir)
                    self.scaler[k] = joblib.load(self._path + dir)

            if k not in self.scaler.keys() or create:
                print(
                    f"[Set] : [{k}] - {StandardScaler()} ",
                )
                self.scaler[k] = StandardScaler()

    def save(self):
        for k in self.scaler.keys():
            joblib.dump(self.scaler[k], self._path + f"scaler_{k}.gz")

    def groupTransform(self, c, data, **kwargs):
        # ------------------ customize --------------------
        if c in kwargs["QUOTE_COL"]:
            return self.scaler["QUOTE"].transform(data)
        elif c in kwargs["VEGA_COL"]:
            return self.scaler["VEGA"].transform(data)
        elif c in kwargs["VOLUME_COL"]:
            return self.scaler["VOLUME"].transform(data)
        else:
            return self.scaler[c].transform(data)
