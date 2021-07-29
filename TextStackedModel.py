# My files
from ClassifierDetails import stop_words
from time import perf_counter
# Packages
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn import preprocessing, naive_bayes, metrics


def load_clean_data():
    train_df = pd.read_csv("CSVs/TrainingImages.csv")
    test_df = pd.read_csv("CSVs/TestingImages.csv")

    # Some images don't have hashtags, mentions, or captions so I need to replace the np.nan values with an empty string
    train_df["Mentions"] = train_df["Mentions"].replace(np.nan, '', regex=True)
    train_df["Caption Lemma"] = train_df["Caption Lemma"].replace(np.nan, '', regex=True)
    print("There are", len(train_df["Mentions"]), "training examples")
    # As there are no weights to update using these classifiers we do not need the validation data
    # Thus, we can just skip to the testing data.
    test_df["Hashtags"] = test_df["Hashtags"].replace(np.nan, '', regex=True)
    test_df["Caption Lemma"] = test_df["Caption Lemma"].replace(np.nan, '', regex=True)
    print("There are", len(test_df["Mentions"]), "testing examples")

    # Encode the targets
    encoder = preprocessing.LabelEncoder()
    # Train labels gets converted to an array of encode values from 1 to N.
    train_labels = encoder.fit_transform(train_df["Class"])
    test_labels = encoder.fit_transform(test_df["Class"])

    return train_df, test_df, train_labels, test_labels


train_df, test_df, train_labels, test_labels = load_clean_data()


# Credit: https://stackoverflow.com/a/63456961/4367851
def compute_count():
    # create a count vectorizer object
    count_vect1 = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words=stop_words)
    count_vect1.fit(train_df['Mentions'])

    # transform the training and test data using count vectorizer object
    Mention_train_count = count_vect1.transform(train_df["Mentions"])
    Mention_test_count = count_vect1.transform(test_df["Mentions"])

    # create a count vectorizer object
    count_vect2 = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words=stop_words)
    count_vect2.fit(train_df['Hashtags'])

    # transform the training and test data using count vectorizer object
    Hash_train_count = count_vect2.transform(train_df["Hashtags"])
    Hash_test_count = count_vect2.transform(test_df["Hashtags"])

    # Captions
    # create a count vectorizer object
    count_vect3 = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words=stop_words)
    count_vect3.fit(train_df['Caption Lemma'])

    # transform the training and test data using count vectorizer object
    Caption_train_count = count_vect3.transform(train_df["Caption Lemma"])
    Caption_test_count = count_vect3.transform(test_df["Caption Lemma"])

    # Stack matrices horizontally (column wise) using hstack().
    train_count = hstack([Mention_train_count, Hash_train_count, Caption_train_count])
    test_count = hstack([Mention_test_count, Hash_test_count, Caption_test_count])

    return train_count, test_count


train_count, test_count = compute_count()


def compute_word_level():
    tfidf_vect1 = TfidfVectorizer(analyzer='word', max_features=5000, stop_words=stop_words)
    tfidf_vect1.fit(train_df['Mentions'])

    # transform the training and test data using count vectorizer object
    Mention_train_word = tfidf_vect1.transform(train_df["Mentions"])
    Mention_test_word = tfidf_vect1.transform(test_df["Mentions"])

    tfidf_vect2 = TfidfVectorizer(analyzer='word', max_features=5000, stop_words=stop_words)
    tfidf_vect2.fit(train_df['Hashtags'])

    # transform the training and test data using count vectorizer object
    Hash_train_word = tfidf_vect2.transform(train_df["Hashtags"])
    Hash_test_word = tfidf_vect2.transform(test_df["Hashtags"])

    tfidf_vect3 = TfidfVectorizer(analyzer='word', max_features=5000, stop_words=stop_words)
    tfidf_vect3.fit(train_df['Caption Lemma'])

    # transform the training and test data using count vectorizer object
    Caption_train_word = tfidf_vect3.transform(train_df["Caption Lemma"])
    Caption_test_word = tfidf_vect3.transform(test_df["Caption Lemma"])

    # Stack matrices horizontally (column wise) using hstack().
    train_word = hstack([Mention_train_word, Hash_train_word, Caption_train_word])
    test_word = hstack([Mention_test_word, Hash_test_word, Caption_test_word])

    return train_word, test_word


train_word, test_word = compute_word_level()


def compute_ngram():
    # Mentions
    # ngram level tf-idf
    tfidf_vect_ngram1 = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), max_features=5000, stop_words="english",
                                        max_df=0.7)
    tfidf_vect_ngram1.fit(train_df['Mentions'])
    Mentions_train_tfidf_ngram = tfidf_vect_ngram1.transform(train_df["Mentions"])
    Mentions_test_tfidf_ngram = tfidf_vect_ngram1.transform(test_df["Mentions"])

    # Hashtags
    # ngram level tf-idf
    tfidf_vect_ngram2 = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), max_features=5000, stop_words="english",
                                        max_df=0.7)
    tfidf_vect_ngram2.fit(train_df['Hashtags'])
    Hashtags_train_tfidf_ngram = tfidf_vect_ngram2.transform(train_df["Hashtags"])
    Hashtags_test_tfidf_ngram = tfidf_vect_ngram2.transform(test_df["Hashtags"])

    # Captions
    # ngram level tf-idf
    tfidf_vect_ngram3 = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), max_features=5000, stop_words="english",
                                        max_df=0.7)
    tfidf_vect_ngram3.fit(train_df['Caption Lemma'])
    Captions_train_tfidf_ngram = tfidf_vect_ngram3.transform(train_df["Caption Lemma"])
    Captions_test_tfidf_ngram = tfidf_vect_ngram3.transform(test_df["Caption Lemma"])

    # Stack matrices horizontally (column wise) using hstack().
    train_ngram = hstack([Mentions_train_tfidf_ngram, Hashtags_train_tfidf_ngram, Captions_train_tfidf_ngram])
    test_ngram = hstack([Mentions_test_tfidf_ngram, Hashtags_test_tfidf_ngram, Captions_test_tfidf_ngram])

    return train_ngram, test_ngram


train_ngram, test_ngram = compute_ngram()


def compute_char():
    # Mentions
    # ngram level tf-idf
    tfidf_vect_char1 = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=5000)
    tfidf_vect_char1.fit(train_df['Mentions'])
    Mentions_train_tfidf_char = tfidf_vect_char1.transform(train_df["Mentions"])
    Mentions_test_tfidf_char = tfidf_vect_char1.transform(test_df["Mentions"])

    # Hashtags
    # ngram level tf-idf
    tfidf_vect_char2 = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=5000)
    tfidf_vect_char2.fit(train_df['Hashtags'])
    Hashtags_train_tfidf_char = tfidf_vect_char2.transform(train_df["Hashtags"])
    Hashtags_test_tfidf_char = tfidf_vect_char2.transform(test_df["Hashtags"])

    # Captions
    # ngram level tf-idf
    tfidf_vect_char3 = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=5000)
    tfidf_vect_char3.fit(train_df['Caption Lemma'])
    Captions_train_tfidf_char = tfidf_vect_char3.transform(train_df["Caption Lemma"])
    Captions_test_tfidf_char = tfidf_vect_char3.transform(test_df["Caption Lemma"])

    # Stack matrices horizontally (column wise) using hstack().
    train_char = hstack([Mentions_train_tfidf_char, Hashtags_train_tfidf_char, Captions_train_tfidf_char])
    test_char = hstack([Mentions_test_tfidf_char, Hashtags_test_tfidf_char, Captions_test_tfidf_char])

    return train_char, test_char


train_char, test_char = compute_char()


#### Create the Model ####
def train_model(level_, classifier, feature_vector_train, label, feature_vector_test, results, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on test dataset
    predictions = classifier.predict(feature_vector_test)  # Vector per classifier
    results[level_] = predictions

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, test_labels)


def fit_classifiers():
    """
    Fits each of the classifiers to the hstacked dataframe that contains the vectorised Hashtags, Mentions, and Captions.
    The
    :return: NA
    """

    results = pd.DataFrame()  # Create blank dataframe
    results["Image_id"] = test_df["Image_id"]
    results["True Label"] = test_labels

    # Naive Bayes on Count Vectors
    nb_count_accuracy = train_model("Count", naive_bayes.MultinomialNB(), train_count, train_labels, test_count, results)
    print("NB, Count Vectors for mentions, tags, and captions, using stacked:", round(nb_count_accuracy, 2))

    # Naive Bayes on Word Level TF IDF Vectors
    nb_tf_accuracy = train_model("WordLevel", naive_bayes.MultinomialNB(), train_word, train_labels, test_word, results)
    print("NB, WordLevel TF-IDF for mentions, tags, and captions, using stacked:", round(nb_tf_accuracy, 2))

    # Naive Bayes on Ngram Level TF IDF Vectors
    nb_ngram_accuracy = train_model("Ngram", naive_bayes.MultinomialNB(), train_ngram, train_labels, test_ngram, results)
    print("NB, N-Gram Vectors for mentions, tags, and captions, using stacked:", round(nb_ngram_accuracy, 2))

    # Naive Bayes on Character Level TF IDF Vectors
    nb_char_accuracy = train_model("Charlevel", naive_bayes.MultinomialNB(), train_char, train_labels, test_char, results)
    print("NB, CharLevel Vectors for mentions, tags, and captions, using stacked: ", round(nb_char_accuracy, 2))

    # Create a prediction from the average of the Naive Bayes Classifiers
    results["NB Average Prediction"] = round(results[["Count", "WordLevel", "Ngram", "Charlevel"]].mean(axis=1), 0)  # Credit: https://stackoverflow.com/a/34735012/4367851
    results.to_csv("CSVs/TextStackedModel.csv", sep=",", index=False)


fit_classifiers()


def average_prediction_accuracy():
    results = pd.read_csv("CSVs/TextStackedModel.csv")
    """
    This function calculates the accuracy of the stacked model.
    This is done by comparing the True Label to the prediction label calculated using the Average Prediction method.
    :param average_type: String, used as a column name
    :return: NA
    """
    matched = 0
    not_matched = 0
    for row in range(len(results)):
        if results.loc[row, "True Label"] == int(results.loc[row, "NB Average Prediction"]):
            matched += 1
        else:
            not_matched += 1

    print("The accuracy of the Text Stacked model when using the Average prediction is:", round(matched / len(results), 2) * 100, "%")


average_prediction_accuracy()


def csv_and_classifier_accuracy():
    # Read in the CSV predictions
    csvresults = pd.read_csv("CSVs/CSVANNPredictions.csv")
    results = pd.read_csv("CSVs/TextStackedModel.csv")

    results["CSVPrediction"] = csvresults["CSV Prediction"]  # Add the CSV results to the TextStackedModel csv as an additional column
    results["Combined Average Prediction"] = results.apply(lambda row: round(row[["NB Average Prediction", "CSVPrediction"]].mean(), 0), axis=1)
    results.to_csv("CSVs/TextSMandCSV.csv", index=False)

    # Calculate the accuracy of combining the ANN and Stacked Text Model
    matched = 0
    not_matched = 0
    for row in range(len(results)):
        if results.loc[row, "True Label"] == int(results.loc[row, "Combined Average Prediction"]):
            matched += 1
        else:
            not_matched += 1

    print("The accuracy of the Text Stacked model and CSVANN when using the Average prediction is:", str(round((matched /len(results))*100, 2)) + "%")


csv_and_classifier_accuracy()
