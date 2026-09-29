import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split, GridSearchCV
from train_all_models import train_all_models_find_the_best_one
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve
import pickle

from sklearn.naive_bayes import GaussianNB

from logcode import setup_logging
from var_transform import transforming_variables
from feature_select import selecting_feature
from imblearn.over_sampling import SMOTE
logger=setup_logging("main")

class HEART_DISEASE_PREDICTION_ML_MODEL:

    def __init__(self,path):
        self.sc_obj = None
        try:
            self.df=pd.read_csv(path)
            logger.info(f"No.of rows and columns in the given excel sheet : {self.df.shape}")
            #self.df=self.df.drop_duplicates()
            logger.info(f"No.of rows and columns after deleting duplicates : {self.df.shape}")
            self.X=self.df.iloc[:,:-1]
            self.Y=self.df.iloc[:,-1]
            self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=0.2,random_state=42)
            logger.info(f"Shape of train data : {self.X_train,self.Y_train}")
            logger.info(f"Shape of test data : {self.X_test,self.Y_test}")
        except Exception as e:
            er_msg,er_ty,er_line=sys.exc_info()
            print(er_msg,er_ty,er_line)

    def clean_data(self):
        try:

            logger.info(f"Checking null values in the data : {self.df.isnull().sum()}")

        except Exception as e:
            er_msg, er_ty, er_line = sys.exc_info()
            print(er_msg, er_ty, er_line)

    def variable_transformation(self):
        try:
           self.X_train,self.X_test=transforming_variables(self.X_train,self.X_test)
           logger.info(f"After variable transformation the first 5 lines of data : {self.X_train.head(5)}")
        except Exception as e:
            er_msg, er_ty, er_line = sys.exc_info()
            print(er_msg, er_ty, er_line)

    def feature_selection(self):
        try:
          self.X_train,self.X_test=selecting_feature(self.X_train,self.X_test,self.Y_train,self.Y_test)
        except Exception as e:
            er_msg, er_ty, er_line = sys.exc_info()
            print(er_msg, er_ty, er_line)

    def balancing_data(self):
        try:
            logger.info(f"Shape before over sampling : {self.Y_train.shape}")
            logger.info(f"Shape of {0} in the data before sampling {sum(self.Y_train==0)}")
            logger.info(f"Shape of {1} in the data before sampling {sum(self.Y_train == 1)}")
            bal_obj=SMOTE(random_state=42)
            self.X_train,self.Y_train=bal_obj.fit_resample(self.X_train,self.Y_train)
            logger.info(f"Shape before over sampling : {self.Y_train.shape}")
            logger.info(f"Shape of {0} in the data before sampling {sum(self.Y_train == 0)}")
            logger.info(f"Shape of {1} in the data before sampling {sum(self.Y_train == 1)}")
        except Exception as e:
            er_msg, er_ty, er_line = sys.exc_info()
            print(er_msg, er_ty, er_line)

    def feature_scaling(self):
       try:
            # Scaling
            self.scal_obj=StandardScaler()
            self.scal_obj.fit(self.X_train)
            self.X_train_balanced_scal=self.scal_obj.transform(self.X_train)
            self.X_test_scal = self.scal_obj.transform(self.X_test)
       except Exception as e:
            er_msg, er_ty, er_line = sys.exc_info()
            print(er_msg, er_ty, er_line)

    def fit_model(self):
        try:
            with open("Heart_Disease_Prediction_Model.pkl","wb") as f:
                pickle.dump(self.scal_obj,f)
        except Exception as e:
            er_msg, er_ty, er_line = sys.exc_info()
            print(er_msg, er_ty, er_line)

    def train_models(self):
        try:
            #train_all_models_find_the_best_one(self.X_train_balanced_scal,self.Y_train,self.X_test_scal,self.Y_test)
            logger.info("------------------------")
            self.m1 = GaussianNB()
            self.m1.fit(self.X_train_balanced_scal, self.Y_train)
            y_pred = self.m1.predict(self.X_test_scal)
            logger.info("Naive Bayes")
            logger.info(f"Accuracy of Naive Bayes : {accuracy_score(self.Y_test, y_pred)}")
            logger.info(f"Confusion Matrix :\n {confusion_matrix(self.Y_test, y_pred)}")
            logger.info(f"Classification Report : {classification_report(self.Y_test, y_pred)}")
            logger.info("------------------------")
            nb_fp, nb_tp, nb_thre = roc_curve(self.Y_test, y_pred)
            # plt.figure(figsize=(5,3))
            # plt.title('AUC & ROC curve')
            # plt.xlabel('flase positive rate')
            # plt.ylabel('true positive rate')
            # plt.plot(nb_fp, nb_tp, label='NB')
            # plt.legend(loc=0)
            # plt.show()
        except Exception as e:
            er_msg, er_ty, er_line = sys.exc_info()
            print(er_msg, er_ty, er_line)


    def hyperparameter_tuning(self):
        try:
            # hyperparameter tuning
            parameters={ "var_smoothing" : np.logspace(-10,-1,10)}
            grid_obj=GridSearchCV(estimator=self.m1,
                                   param_grid=parameters,
                                   cv=5,
                                   scoring='accuracy',
                                   n_jobs=-1)
            grid_obj.fit(self.X_train_balanced_scal,self.Y_train)
            logger.info(f'parameters are:{grid_obj.best_params_}')
            logger.info(f'accuracy:{grid_obj.best_score_}')
        except Exception as e:
            er_msg, er_ty, er_line = sys.exc_info()
            print(er_msg, er_ty, er_line)

    def testing(self):
        try:
            # ---------------- TESTING ----------------

            testing = np.array([[33, 1, 3, 169, 0.7, 0, 2]])

            # Convert custom input to DataFrame
            testing_df = pd.DataFrame(
                testing,
                columns=self.X_train.columns
            )

            # Apply the same scaler used during training
            testing_scaled = self.scal_obj.transform(testing_df)

            # Prediction
            prediction = self.m1.predict(testing_scaled)

            logger.info(f"Custom input: {testing}")
            logger.info(f"Scaled custom input: {testing_scaled}")
            logger.info(f"Prediction of Naive Bayes model: {prediction[0]}")

            # save the model
            with open('navi_bayes_model.pkl', 'wb') as b:
                pickle.dump(self.m1, b)


        except Exception as e:
            er_msg, er_ty, er_line = sys.exc_info()
            print(er_msg, er_ty, er_line)

if __name__ == '__main__':
    obj=HEART_DISEASE_PREDICTION_ML_MODEL("heart.csv")
    obj.clean_data()
    obj.variable_transformation()
    obj.balancing_data()
    obj.feature_selection()
    obj.feature_scaling()
    obj.fit_model()
    obj.train_models()
    obj.hyperparameter_tuning()
    obj.testing()

