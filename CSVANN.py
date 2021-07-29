import pandas as pd
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from tensorflow import keras
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from Plots import plot_importances, plot_conf_matrix, plot_loss, plot_accuracy
from sklearn.model_selection import cross_validate
import tensorflow as tf


def load_clean_data():
    training_data = pd.read_csv("CSVs/TrainingImages.csv")
    validation_data = pd.read_csv("CSVs/ValidationImages.csv")
    test_data = pd.read_csv("CSVs/TestingImages.csv")
    cross_val_data = pd.read_csv("CSVs/UsersImages.csv")

    training_data[["Saturation", "Hue"]] = training_data[["Saturation", "Hue"]].fillna("0")
    validation_data[["Saturation", "Hue"]] = validation_data[["Saturation", "Hue"]].fillna("0")
    test_data[["Saturation", "Hue"]] = test_data[["Saturation", "Hue"]].fillna("0")
    cross_val_data[["Saturation", "Hue"]] = cross_val_data[["Saturation", "Hue"]].fillna("0")
    return training_data, validation_data, test_data, cross_val_data


training_data, validation_data, test_data, cross_val_data = load_clean_data()


columns_list = ["Hashtag Count", "Mention Count", "DaysSinceLastPosting", "Caption Length", "Saturation", "Hue",
                "CaptionTruncated"]


def standardise_columns(columns_list=columns_list):
    # Standardise columns
    training_data[columns_list] = \
        StandardScaler().fit_transform(np.array(training_data[columns_list]))

    validation_data[columns_list] = \
        StandardScaler().fit_transform(np.array(validation_data[columns_list]))

    test_data[columns_list] = \
        StandardScaler().fit_transform(np.array(test_data[columns_list]))

    cross_val_data[columns_list] = \
        StandardScaler().fit_transform(np.array(cross_val_data[columns_list]))

    return training_data, validation_data, test_data, cross_val_data


def encode_data():
    training_data, validation_data, test_data, cross_val_data = standardise_columns(columns_list)
    # Training data
    # We need to convert the Inputs and labels to arrays to be then fed into the model.
    Input = training_data[columns_list].values
    print("There are", Input.shape[0], "training examples")
    Input = np.asarray(Input).astype(np.float32)

    Labels = training_data["Class"].values
    Labels = np.asarray(Labels).astype(np.float32).reshape((-1, 1))
    n_classes = len(np.unique(Labels))

    # Validation data
    Validation_input = validation_data[columns_list].values
    print("There are", Validation_input.shape[0], "validation examples")
    Validation_input = np.asarray(Validation_input).astype(np.float32)

    # Testing data
    Test_input = test_data[columns_list].values
    Test_input = np.asarray(Test_input).astype(np.float32)
    print("There are", Test_input.shape[0], "testing examples")

    # Cross Validation data
    CV_input = cross_val_data[columns_list].values
    CV_input = np.asarray(CV_input).astype(np.float32)

    return Input, Labels, Validation_input, Test_input, CV_input, n_classes


Input, Labels, Validation_input, Test_input, CV_input, n_classes = encode_data()


def encode_labels():
    # Encode the targets
    encoder = preprocessing.LabelEncoder()
    # Train labels gets converted to an array of encode values from 1 to N.
    Training_Labels = encoder.fit_transform(training_data["Class"])
    Validation_labels = encoder.fit_transform(validation_data["Class"])
    Test_Labels = encoder.fit_transform(test_data["Class"])
    CV_labels = encoder.fit_transform(cross_val_data["Class"])

    return Training_Labels, Validation_labels, Test_Labels, CV_labels


Training_Labels, Validation_labels, Test_Labels, CV_labels = encode_labels()


### XGBClassifier ###
def xgb_classifier(tune="True"):
    from xgboost import XGBClassifier
    if tune == "True":
        ### Random Search ###
        # Credit: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        from sklearn.model_selection import RandomizedSearchCV
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        gamma = [float(x) for x in np.linspace(0, 1, num=20)]
        eta = [float(x) for x in np.linspace(0, 1, num=20)]
        reg_alpha = [float(x) for x in np.linspace(0, 1, num=20)]
        reg_lambda = [float(x) for x in np.linspace(0, 1, num=20)]
        learning_rate = [float(x) for x in np.linspace(0, 1, num=20)]

        random_grid = {'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       "gamma": gamma,
                       "learning_rate": learning_rate,
                       "reg_alpha": reg_alpha,
                       "reg_lambda": reg_lambda,
                       "eta": eta}

        # Fit the base model
        clf = XGBClassifier(use_label_encoder= False, max_depth=5, objective="multi:softprob", eval_metric="mlogloss")

        clf = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=1, cv=3, verbose=2,
                                 random_state=123, n_jobs=-1)  # Fit the random search model
        clf.fit(Input, Labels)

        y_pred = clf.predict(Test_input)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(Test_Labels, predictions)
        print(clf.get_params())
        print("XGBClassification accuracy (Tuned): %.2f%%" % (accuracy * 100.0))

    else:
        clf = XGBClassifier(use_label_encoder=False, max_depth=40, n_estimators=1000, learning_rate=0.842, gamma=0.421,
                            reg_alpha=0.579, reg_lambda=0.105,  objective="multi:softprob", eval_metric="mlogloss")
        clf.fit(Input, Training_Labels)
        plot_importances(clf, Input, columns_list, "(XGBoost - Tuned)")
        y_pred = clf.predict(Test_input)
        predictions = [round(value) for value in y_pred]
        accuracy_tuned = accuracy_score(Test_Labels, predictions)
        print("XGBClassification accuracy (Tuned): %.2f%%" % (accuracy_tuned*100.0))

    clf_base = XGBClassifier(use_label_encoder=False, objective="multi:softprob", eval_metric="mlogloss")
    clf_base.fit(Input, Training_Labels)
    plot_importances(clf_base, Input, columns_list, "(XGBoost - Base)")
    y_pred = clf_base.predict(Test_input)
    predictions = [round(value) for value in y_pred]
    accuracy_base = accuracy_score(Test_Labels, predictions)
    print("XGBClassification accuracy (Untuned): %.2f%%" % (accuracy_base*100.0))

    if accuracy_base > accuracy_tuned:
        return clf_base, accuracy_base
    else:
        return clf, accuracy_tuned


def random_for(tune="False"):
    from sklearn.ensemble import RandomForestClassifier

    if tune == "True":
        ### Random Search ###
        # Credit: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]  # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'bootstrap': bootstrap}


        # Fit the base model
        clf = RandomForestClassifier(max_depth = 5)

        clf = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100, cv=3, verbose=0,
                                 random_state=123, n_jobs=-1)  # Fit the random search model
        clf.fit(Input, Training_Labels)

        y_pred = clf.predict(Test_input)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(Test_Labels, predictions)
        #print(clf.get_params())
        print("Random Forest accuracy (Tuned): %.2f%%" % (accuracy*100.0))

    else:
        clf = RandomForestClassifier(max_depth=5, bootstrap= True, ccp_alpha= 0, class_weight= None, criterion= "gini",
                                     max_features="auto", max_leaf_nodes= None, min_samples_leaf= 1, min_samples_split=2)
        clf.fit(Input, Training_Labels)
        plot_importances(clf, Input, columns_list, "(Random Forest - Tuned)")

        y_pred = clf.predict(Test_input)
        predictions = [round(value) for value in y_pred]
        accuracy_tuned = accuracy_score(Test_Labels, predictions)
        print("Random Forest accuracy (Tuned): %.2f%%" % (accuracy_tuned*100.0))

    clf_base = RandomForestClassifier()
    clf_base.fit(Input, Training_Labels)
    plot_importances(clf_base, Input, columns_list, "(Random Forest - Base)")

    y_pred = clf_base.predict(Test_input)
    predictions = [round(value) for value in y_pred]
    accuracy_base = accuracy_score(Test_Labels, predictions)
    print("Random Forest accuracy (Untuned): %.2f%%" % (accuracy_base * 100.0))

    rf = RandomForestClassifier()
    cv_results = cross_validate(rf, CV_input, CV_labels, cv=5)
    print("Random Forest accuracy (Cross Validation) %.2f%%" % (cv_results['test_score'].mean()*100))

    if accuracy_base > accuracy_tuned:
        return clf_base, accuracy_base
    else:
        return clf, accuracy_tuned


def dtree(tune="False"):
    from sklearn.tree import DecisionTreeClassifier

    if tune == "True":
        ### Random Search ###
        # Credit: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt', "log2"]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Maximum number of leaf nodes
        max_leaf_nodes = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [2, 5, 10]
        # Splitter type
        splitter = ["best", "random"]

        # Method of selecting samples for training each tree
        bootstrap = [True, False]  # Create the random grid
        random_grid = {'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_leaf_nodes': max_leaf_nodes,
                       'splitter': splitter}

        # Fit the base model
        clf = DecisionTreeClassifier()

        clf = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100, cv=3, verbose=0,
                                 random_state=123, n_jobs=-1)  # Fit the random search model
        clf.fit(Input, Training_Labels)

        y_pred = clf.predict(Test_input)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(Test_Labels, predictions)
        print("Decision Tree accuracy (Tuned): %.2f%%" % (accuracy*100.0))

    else:
        clf = DecisionTreeClassifier(max_depth=5, ccp_alpha=0, class_weight=None, criterion="gini", min_samples_split=2, splitter="best")
        clf.fit(Input, Training_Labels)

        plot_importances(clf, Input, columns_list, "(Decision Tree - Tuned)")

        y_pred = clf.predict(Test_input)
        predictions = [round(value) for value in y_pred]
        accuracy_tuned = accuracy_score(Test_Labels, predictions)
        print("Decision Tree accuracy (Tuned): %.2f%%" % (accuracy_tuned*100.0))

    clf_base = DecisionTreeClassifier()
    clf_base.fit(Input, Training_Labels)
    plot_importances(clf_base, Input, columns_list, "(Decision Tree - Base)")
    y_pred = clf_base.predict(Test_input)
    predictions = [round(value) for value in y_pred]
    accuracy_base = accuracy_score(Test_Labels, predictions)
    print("Decision Tree accuracy (Untuned): %.2f%%" % (accuracy_base*100.0))

    dtc = DecisionTreeClassifier()
    cv_results = cross_validate(dtc, CV_input, CV_labels, cv=5)
    print("Decision Tree accuracy (Cross Validation) %.2f%%" % (cv_results['test_score'].mean()*100))

    # Decision tree using only the important variables
    clf_base = DecisionTreeClassifier()
    clf_base.fit(Input, Training_Labels)
    plot_importances(clf_base, Input, columns_list, "(Decision Tree - Base)")
    y_pred = clf_base.predict(Test_input)
    predictions = [round(value) for value in y_pred]
    accuracy_base = accuracy_score(Test_Labels, predictions)
    print("Decision Tree accuracy (important variables) : %.2f%%" % (accuracy_base*100.0))

    if accuracy_base > accuracy_tuned:
        return clf_base, accuracy_base
    else:
        return clf, accuracy_tuned


def adatree(tune="True"):
    from sklearn.ensemble import AdaBoostClassifier

    if tune == "True":
        ### Random Search ###
        # Credit: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
        algorithm = ["SAMME", "SAMME.R"]
        learning_rate = [float(x) for x in np.linspace(0, 1, num=10)]
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]

        random_grid = {"algorithm": algorithm,
                       "learning_rate": learning_rate,
                       "n_estimators": n_estimators}

        clf = AdaBoostClassifier()

        clf = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100, cv=3, verbose=1,
                                 random_state=123, n_jobs=-1)  # Fit the random search model
        clf.fit(Input, Training_Labels)
        y_pred = clf.predict(Test_input)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(Test_Labels, predictions)
        #print(clf.get_params())
        print("AdaBoost Classifier accuracy (Tuned): %.2f%%" % (accuracy * 100.0))

    else:
        clf = AdaBoostClassifier(algorithm="SAMME.R", learning_rate=1, n_estimators=50)

        clf.fit(Input, Training_Labels)
        plot_importances(clf, Input, columns_list, "(AdaBoost - Tuned)")
        y_pred = clf.predict(Test_input)
        predictions = [round(value) for value in y_pred]
        accuracy_tuned = accuracy_score(Test_Labels, predictions)
        print("AdaBoost Classifier accuracy (Tuned): %.2f%%" % (accuracy_tuned * 100.0))

    clf_base = AdaBoostClassifier()
    clf_base.fit(Input, Training_Labels)
    plot_importances(clf_base, Input, columns_list, "(AdaBoost - Base)")

    y_pred = clf_base.predict(Test_input)
    predictions = [round(value) for value in y_pred]
    accuracy_base = accuracy_score(Test_Labels, predictions)
    print("AdaBoost Classifier accuracy (Untuned): %.2f%%" % (accuracy_base*100.0))

    ada_cv_classifier = AdaBoostClassifier()
    cv_results = cross_validate(ada_cv_classifier, CV_input, CV_labels, cv=5)
    print("AdaBoost Classifier accuracy (Cross Validation) %.2f%%" % (cv_results['test_score'].mean()*100))

    if accuracy_base > accuracy_tuned:
        return clf_base, accuracy_base
    else:
        return clf, accuracy_tuned

# I decided against Logistic regression, but have left the code here for future use.
def logistic_reg():
    clf = LogisticRegression(random_state=123, max_iter=400).fit(Input, Training_Labels)
    clf.predict(Test_input)
    score = clf.score(Input, Training_Labels)
    print("Logistic Regression Accuracy with no penalty: %.2f%%" % (score*100.0))


def logistic_reg_l1():
    clf = LogisticRegression(fit_intercept=True, random_state=123, max_iter=400, penalty="l1", solver="liblinear").fit(Input, Training_Labels)
    clf.predict(Test_input)
    score = clf.score(Input, Training_Labels)
    print("Logistic Regression Accuracy with L1 penalty: %.2f%%" % (score*100.0))

    clf_ni = LogisticRegression(fit_intercept=False, random_state=123, max_iter=400, penalty="l1", solver="liblinear").fit(Input, Training_Labels)
    clf_ni.predict(Test_input)
    score = clf_ni.score(Input, Training_Labels)
    print("Logistic Regression Accuracy with L1 penalty and no intercept: %.2f%%" % (score * 100.0))


def logistic_reg_l2():
    clf = LogisticRegression(fit_intercept=True, random_state=123, max_iter=400, penalty="l2").fit(Input, Training_Labels)
    clf.predict(Test_input)
    score = clf.score(Input, Training_Labels)
    print("Logistic Regression Accuracy with L2 penalty: %.2f%%" % (score*100.0))

    clf_ni = LogisticRegression(fit_intercept=False, random_state=123, max_iter=400, penalty="l2").fit(Input, Training_Labels)
    clf_ni.predict(Test_input)
    score = clf_ni.score(Input, Training_Labels)
    print("Logistic Regression Accuracy with L2 penalty and no intercept: %.2f%%" % (score*100.0))

    clf_bal = LogisticRegression(fit_intercept=True, class_weight="balanced", random_state=123, max_iter=400, penalty="l2").fit(Input, Training_Labels)
    clf_bal.predict(Test_input)
    score = clf_bal.score(Input, Training_Labels)
    print("Logistic Regression Accuracy with L2 penalty, balanced classes: %.2f%%" % (score*100.0))


# normalize a vector to have unit norm
def normalize(weights):
    from numpy.linalg import norm
    # calculate l1 vector norm
    result = norm(weights, 1)
    # check for a vector of all zeros
    if result == 0.0:
        return weights
    # return normalized vector (unit norm)
    return weights / result


# Credit: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html
def voting_class(xclf, rfclf, dclf, adaclf, xgb_accuracy, randomfor_accuracy, decision_tree_accuracy, adaboost_accuracy):
    voting_types = ["soft", "hard"]

    # Equal weights
    for vt in voting_types:
        clf = VotingClassifier(estimators=[("rf", rfclf), ("ada", adaclf), ("dclf", dclf), ("xclf", xclf)], voting=vt)
        clf.fit(Input, Training_Labels)
        y_pred = clf.predict(Test_input)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(Test_Labels, predictions)
        print("Ensemble Classifier accuracy using", str(vt).capitalize(), "voting: %.2f%%" % (accuracy*100.0))

    # Advantage weighting
    for vt in voting_types:
        base = min(xgb_accuracy, randomfor_accuracy, decision_tree_accuracy, adaboost_accuracy)
        clf = VotingClassifier(estimators=[("rf", rfclf), ("ada", adaclf), ("dclf", dclf), ("xclf", xclf)],
                               weights=[xgb_accuracy-base, randomfor_accuracy-base, decision_tree_accuracy-base,
                                        adaboost_accuracy-base], voting=vt)
        clf.fit(Input, Training_Labels)
        y_pred = clf.predict(Test_input)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(Test_Labels, predictions)
        print("Ensemble Classifier accuracy using", str(vt).capitalize(), "voting: %.2f%%" % (accuracy * 100.0),
              "and proportional weights")

    # Accuracy weights
    for vt in voting_types:
        clf = VotingClassifier(estimators=[("rf", rfclf), ("ada", adaclf), ("dclf", dclf), ("xclf", xclf)],
                               weights=[xgb_accuracy, randomfor_accuracy, decision_tree_accuracy,
                                        adaboost_accuracy], voting=vt)
        clf.fit(Input, Training_Labels)
        y_pred = clf.predict(Test_input)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(Test_Labels, predictions)
        print("Ensemble Classifier accuracy using", str(vt).capitalize(), "voting: %.2f%%" % (accuracy * 100.0),
              "and Estimated weights")

    # Normalised weights
    for vt in voting_types:
        w1, w2, w3, w4 = normalize([xgb_accuracy, randomfor_accuracy, decision_tree_accuracy, adaboost_accuracy])
        clf = VotingClassifier(estimators=[("rf", rfclf), ("ada", adaclf), ("dclf", dclf), ("xclf", xclf)],
                               weights=[w1, w2, w3, w4], voting=vt)
        clf.fit(Input, Training_Labels)
        y_pred = clf.predict(Test_input)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(Test_Labels, predictions)
        print("Ensemble Classifier accuracy using", str(vt).capitalize(), "voting: %.2f%%" % (accuracy * 100.0),
              "and Normalised weights")


def create_model():
    from keras.utils.vis_utils import plot_model
    model = Sequential()
    model.add(Dense(32, input_dim=Input.shape[1], activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation="relu"))
    model.add(Dense(20, activation="softmax"))
    print(model.summary())

    return model


def train_model(model):
    with tf.device('/device:GPU:0'):
        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=Adam(lr=0.0001, decay=1e-4), metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)
        history = model.fit(Input, Training_Labels, validation_data=(Validation_input, Validation_labels), epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)

        plot_loss(history, type_="ANN")
        plot_accuracy(history, type_="ANN")

        # model.summary()

        loss, accuracy = model.evaluate(Test_input, Test_Labels)
        print("Neural Network Test Accuracy: %.2f%%" % (accuracy*100.0))

        # To get the actual predictions for our test data we must use model.predict()
        # This will be used in the ensemble method.
        names = pd.Series(test_data["Image_id"])
        results_probs = model.predict(Test_input)
        cols = list(range(1, results_probs.shape[1]+1, 1))
        cols.insert(0, "Image_id")
        cols.insert(len(cols), "True Label")
        probabilities = np.column_stack([np.array(names), results_probs, Test_Labels])
        pd.DataFrame(probabilities, columns=[cols]).to_csv("CSVs/CSVANNProbabilities.csv", sep=",", index=False)

        results = np.argmax(results_probs, axis=1)
        predictions = np.column_stack((np.array(names), results))
        pd.DataFrame(predictions, columns=["Image_id", "CSV Prediction"]).to_csv("CSVs/CSVANNPredictions.csv", sep=",", index=False)

        plot_conf_matrix(Test_Labels, results, type_="ANN")


def funcs():
    xclf, xgb_accuracy = xgb_classifier(tune="False")
    rfclf, randomfor_accuracy = random_for(tune="False")
    dclf, decision_tree_accuracy = dtree(tune="False")
    adaclf, adaboost_accuracy = adatree(tune="False")
    #logistic_reg()
    #logistic_reg_l1()
    #logistic_reg_l2()
    voting_class(xclf, rfclf, dclf, adaclf, xgb_accuracy, randomfor_accuracy, decision_tree_accuracy, adaboost_accuracy)
    #model = create_model()
    #train_model(model)

funcs()
