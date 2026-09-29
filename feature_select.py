import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
from logcode import setup_logging
logger=setup_logging("feature_selection")

def selecting_feature(X_train,X_test,Y_train,Y_test):
    # Constant
    constant_obj=VarianceThreshold(threshold=0.0)
    constant_obj.fit(X_train)

    logger.info(f"after doing constant feature selection : {X_train.columns[~constant_obj.get_support()]}")
    X_train=X_train.drop('fbs_yeo_trim',axis=1)
    X_test = X_test.drop('fbs_yeo_trim', axis=1)

    # Quesi Constant
    quesi_constant_obj=VarianceThreshold(threshold=0.1)
    quesi_constant_obj.fit(X_train)
    logger.info(f"after doing quesi constant feature selection : {X_train.columns[~quesi_constant_obj.get_support()]}")
    X_train=X_train.drop(['trestbps_yeo_trim', 'chol_yeo_trim', 'exang_yeo_trim', 'ca_yeo_trim'],axis=1)
    X_test = X_test.drop(['trestbps_yeo_trim', 'chol_yeo_trim', 'exang_yeo_trim', 'ca_yeo_trim'], axis=1)

    #Hypothesis Testing
    # coeff_p_value=[]
    # for i in X_train.columns:
    #     values=pearsonr(X_train[i],Y_train)
    #     coeff_p_value.append(values)
    # coeff_p_value=np.array(coeff_p_value)
    # p_values = coeff_p_value[:, 1]
    #
    # plt.figure(figsize=(5,3))
    # plt.title("Hypothesis Testing")
    #
    # plt.xlabel("column Names")
    # plt.ylabel("P_values for Each Independent column")
    #
    # plt.bar(X_train.columns, p_values)
    #
    # plt.show()

    X_train=X_train.drop("restecg_yeo_trim",axis=1)
    X_test = X_test.drop("restecg_yeo_trim", axis=1)
    return X_train,X_test