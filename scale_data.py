import pandas as pd
import Consts

class ScaleData:
    scale_args = dict()

    def scale_train(self, df: pd.DataFrame):
        scaled = "_scaled"
        for feature in df.keys():
            t = df[feature].describe().transpose()
            if feature in Consts.setGaussianFeatures:
                miu, sigma = t["mean"], t["std"]
                df[feature + scaled] = (df[feature] - miu) / sigma
                self.scale_args[feature] = (miu, sigma)

            elif feature in Consts.setUniformFeatures:
                min_val, max_val = t["min"], t["max"]
                df[feature + scaled] = (df[feature] - min_val) * 2 / (max_val - min_val) + min_val
                self.scale_args[feature] = (min_val, max_val)

    def scale_test(self, df: pd.DataFrame):
        scaled = "_scaled"
        for feature in df.keys():
            if feature in Consts.setGaussianFeatures:
                miu, sigma = self.scale_args[feature]
                df[feature + scaled] = (df[feature] - miu) / sigma

            elif feature in Consts.setUniformFeatures:
                min_val, max_val = self.scale_args[feature]
                df[feature + scaled] = (df[feature] - min_val) * 2 / (max_val - min_val) + min_val
