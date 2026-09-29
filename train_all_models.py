import sys
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from logcode import setup_logging
logger=setup_logging("Training_all_the_models")

def train_all_models_find_the_best_one(X_train,Y_train,X_test,Y_test):
    try:
        logger.info("------------------------")
        m1=LogisticRegression()
        m1.fit(X_train,Y_train)
        y_pred=m1.predict(X_test)
        logger.info("Logistic Regression")
        logger.info(f"Accuracy of Logistic Regression Model : {accuracy_score(Y_test,y_pred)}")
        logger.info(f"Confusion Matrix :\n {confusion_matrix(Y_test,y_pred)}")
        logger.info(f"Classification Report : {classification_report(Y_test,y_pred)}")
        logger.info("------------------------")


        m2 = KNeighborsClassifier(n_neighbors=5)
        m2.fit(X_train, Y_train)
        y_pred = m2.predict(X_test)
        logger.info("KNN")
        logger.info(f"Accuracy of KNN Model : {accuracy_score(Y_test, y_pred)}")
        logger.info(f"Confusion Matrix :\n {confusion_matrix(Y_test, y_pred)}")
        logger.info(f"Classification Report : {classification_report(Y_test, y_pred)}")
        logger.info("------------------------")

        m3 = DecisionTreeClassifier()
        m3.fit(X_train, Y_train)
        y_pred = m3.predict(X_test)
        logger.info("DecisionTree")
        logger.info(f"Accuracy of Decision Tree : {accuracy_score(Y_test, y_pred)}")
        logger.info(f"Confusion Matrix :\n {confusion_matrix(Y_test, y_pred)}")
        logger.info(f"Classification Report : {classification_report(Y_test, y_pred)}")
        logger.info("------------------------")

        m4 = RandomForestClassifier()
        m4.fit(X_train, Y_train)
        y_pred = m4.predict(X_test)
        logger.info("Random Forest")
        logger.info(f"Accuracy of Random Forest : {accuracy_score(Y_test, y_pred)}")
        logger.info(f"Confusion Matrix :\n {confusion_matrix(Y_test, y_pred)}")
        logger.info(f"Classification Report : {classification_report(Y_test, y_pred)}")
        logger.info("------------------------")

        # m5 = SVC()
        # m5.fit(X_train, Y_train)
        # y_pred = m5.predict(X_test)
        # logger.info("SVM")
        # logger.info(f"Accuracy of SVM : {accuracy_score(Y_test, y_pred)}")
        # logger.info(f"Confusion Matrix :\n {confusion_matrix(Y_test, y_pred)}")
        # logger.info(f"Classification Report : {classification_report(Y_test, y_pred)}")
        # logger.info("------------------------")

        m6 = GaussianNB()
        m6.fit(X_train, Y_train)
        y_pred = m6.predict(X_test)
        logger.info("Naive Bayes")
        logger.info(f"Accuracy of Naive Bayes : {accuracy_score(Y_test, y_pred)}")
        logger.info(f"Confusion Matrix :\n {confusion_matrix(Y_test, y_pred)}")
        logger.info(f"Classification Report : {classification_report(Y_test, y_pred)}")
        logger.info("------------------------")

        m7 = GradientBoostingClassifier()
        m7.fit(X_train, Y_train)
        y_pred = m7.predict(X_test)
        logger.info("Gradient Boosting")
        logger.info(f"Accuracy of Gradient Boosting : {accuracy_score(Y_test, y_pred)}")
        logger.info(f"Confusion Matrix :\n {confusion_matrix(Y_test, y_pred)}")
        logger.info(f"Classification Report : {classification_report(Y_test, y_pred)}")
        logger.info("------------------------")

        m8 = AdaBoostClassifier()
        m8.fit(X_train, Y_train)
        y_pred = m8.predict(X_test)
        logger.info("AdaBoosting")
        logger.info(f"Accuracy of AdaBoosting : {accuracy_score(Y_test, y_pred)}")
        logger.info(f"Confusion Matrix :\n {confusion_matrix(Y_test, y_pred)}")
        logger.info(f"Classification Report : {classification_report(Y_test, y_pred)}")
        logger.info("------------------------")

        m9 = XGBClassifier()
        m9.fit(X_train, Y_train)
        y_pred = m9.predict(X_test)
        logger.info("XGBoosting")
        logger.info(f"Accuracy of XGBoosting : {accuracy_score(Y_test, y_pred)}")
        logger.info(f"Confusion Matrix :\n {confusion_matrix(Y_test, y_pred)}")
        logger.info(f"Classification Report : {classification_report(Y_test, y_pred)}")
        logger.info("------------------------")

        knn_predictions=m2.predict(X_test)
        nb_predictions = m6.predict(X_test)
        lr_predictions = m1.predict(X_test)
        dt_predictions = m3.predict(X_test)
        rf_predictions = m4.predict(X_test)
        ab_predictions = m8.predict(X_test)
        gb_predictions = m7.predict(X_test)
        xgb_predictions = m9.predict(X_test)


        knn_fp,knn_tp,knn_thre=roc_curve(Y_test,knn_predictions)
        nb_fp, nb_tp, nb_thre = roc_curve(Y_test, nb_predictions)
        lr_fp, lr_tp, lr_thre = roc_curve(Y_test, lr_predictions)
        dt_fp, dt_tp, dt_thre = roc_curve(Y_test, dt_predictions)
        rf_fp, rf_tp, rf_thre = roc_curve(Y_test, rf_predictions)
        ab_fp, ab_tp, ab_thre = roc_curve(Y_test, ab_predictions)
        gb_fp, gb_tp, gb_thre = roc_curve(Y_test, gb_predictions)
        xgb_fp, xgb_tp, xgb_thre = roc_curve(Y_test, xgb_predictions)

        plt.figure(figsize=(5,3))
        plt.title('AUC & ROC curve')
        plt.xlabel('flase positive rate')
        plt.ylabel('true positive rate')

        plt.plot(knn_fp,knn_tp,label='KNN')
        plt.plot(nb_fp, nb_tp, label='NB')
        plt.plot(lr_fp, lr_tp, label='LR')
        plt.plot(dt_fp, dt_tp, label='DT')
        plt.plot(rf_fp, rf_tp, label='RF')
        plt.plot(ab_fp, ab_tp, label='AB')
        plt.plot(gb_fp, gb_tp, label='GB')
        plt.plot(xgb_fp, xgb_tp, label='XGB')
        plt.legend(loc=0)
        plt.show()

    except Exception as e:
        er_ty,er_msg,er_line=sys.exc_info()