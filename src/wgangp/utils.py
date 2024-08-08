def debug(x):
    print("===== [DEBUG] =====")
    if isinstance(x, list):
        for i in x:
            print(i)
    else:
        print(x)


from sklearn.preprocessing import MinMaxScaler

import joblib
import os


class Scaler:
    def __init__(self, keys, path="./data/scaler/", scale=MinMaxScaler()):
        self.scaler = {}
        self._scale = scale
        self._keys = keys
        self._path = path
        self._creatScaler()

    def __call__(self):
        return self.scaler

    @property
    def keys(self):
        return self._keys

    def _creatScaler(self):
        if not os.path.exists(self._path):
            os.makedirs(self._path)
        for k in self._keys:
            self.scaler[k] = self._scale

    def save(self):
        for k in self.scaler.keys():
            joblib.dump(self.scaler[k], self._path + f"scaler_{k}.gz")

    def loads(self):
        dir_list = os.listdir(self._path)
        for dir in dir_list:
            key = dir.split("scaler_")[-1].replace(".gz", "")
            self._scale[key] = joblib.load(dir)
