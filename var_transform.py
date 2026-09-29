import sys
import numpy as np
from scipy.stats import yeojohnson
from logcode import setup_logging
logger=setup_logging("var_transform")

def transforming_variables(X_train,X_test):
    try:

        for i in X_train.columns:
            X_train[i + '_yeo'], lambda_val = yeojohnson(X_train[i])
            X_test[i + '_yeo'], lambda_val = yeojohnson(X_test[i])
            X_train = X_train.drop(i, axis=1)
            X_test = X_test.drop(i, axis=1)

            iqr = X_train[i + "_yeo"].quantile(0.75) - X_train[i + "_yeo"].quantile(0.25)
            upper_limit = X_train[i + "_yeo"].quantile(0.75) + (1.5 * iqr)
            lower_limit = X_train[i + "_yeo"].quantile(0.25) - (1.5 * iqr)

            X_train[i + '_yeo_trim'] = np.where(X_train[i+ "_yeo"] > upper_limit, upper_limit,
                                            np.where(X_train[i+ "_yeo"] < lower_limit, lower_limit, X_train[i+ "_yeo"]))
            X_test[i + '_yeo_trim'] = np.where(X_test[i+ "_yeo"] > upper_limit, upper_limit,
                                           np.where(X_test[i+ "_yeo"] < lower_limit, lower_limit, X_test[i+ "_yeo"]))
            X_train = X_train.drop(i+ "_yeo", axis=1)
            X_test = X_test.drop(i+ "_yeo", axis=1)
        logger.info(f"After Variable Transformation, The first 5 records : {X_train.head(5)}")
        return X_train,X_test
    except Exception as e:
        er_msg, er_ty, er_line = sys.exc_info()
        print(er_msg, er_ty, er_line)