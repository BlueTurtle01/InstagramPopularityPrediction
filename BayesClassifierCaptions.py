import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import pandas as pd
import numpy as np
import nltk; nltk.download('wordnet')
from sklearn import preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from ClassifierDetails import stop_words

# Credit: https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/


def load_data():
    train_df = pd.read_csv("CSVs/TrainingImages.csv")
    train_df["Caption Short"] = train_df["Caption Short"].replace(np.nan, '', regex=True)
    print("There are", len(train_df["Caption Short"]), "training examples")

    validation_df = pd.read_csv("CSVs/ValidationImages.csv")
    validation_df["Caption Short"] = validation_df["Caption Short"].replace(np.nan, '', regex=True)
    print("There are", len(validation_df["Caption Short"]), "validation examples")

    test_df = pd.read_csv("CSVs/TestingImages.csv")
    test_df["Caption Short"] = test_df["Caption Short"].replace(np.nan, '', regex=True)
    print("There are", len(test_df["Caption Short"]), "test examples")
    return train_df, test_df, validation_df


train_df, test_df, validation_df = load_data()


def remove_punct(text):
    ### Remove Punctuation ###
    import string

    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct


def lemma_(train_df, validation_df, test_df):
    # Credit: https://www.machinelearningplus.com/nlp/lemmatization-examples-python/#introduction
    import nltk
    from nltk.stem import WordNetLemmatizer
    nltk.download('punkt')

    lemmatizer = WordNetLemmatizer()

    def l(row):
        word_list = nltk.word_tokenize(row)
        return ' '.join([lemmatizer.lemmatize(w) for w in word_list])

    train_df["Caption Lemma"] = train_df.apply(lambda row: l(row["Caption Short"]), axis=1)
    train_df.to_csv("CSVs/TrainingImages.csv", index=False)

    validation_df["Caption Lemma"] = validation_df.apply(lambda row: l(row["Caption Short"]), axis=1)
    validation_df.to_csv("CSVs/ValidationImages.csv", index=False)

    test_df["Caption Lemma"] = test_df.apply(lambda row: l(row["Caption Short"]), axis=1)
    test_df.to_csv("CSVs/TestingImages.csv", index=False)

    return train_df, validation_df, test_df


def target_encoder(train_df, validation_df):
    # Encode the targets
    encoder = preprocessing.LabelEncoder()
    # Train labels gets converted to an array of encode values from 1 to N.
    train_labels = encoder.fit_transform(train_df["Class"])
    validation_labels = encoder.fit_transform(validation_df["Class"])
    test_labels = encoder.fit_transform(test_df["Class"])

    return train_labels, validation_labels, test_labels


# Remove Punctuation
train_df["Caption Short"] = train_df["Caption Short"].apply(lambda x: remove_punct(x))
validation_df["Caption Short"] = validation_df["Caption Short"].apply(lambda x: remove_punct(x))
test_df["Caption Short"] = test_df["Caption Short"].apply(lambda x: remove_punct(x))

# Lemmatize
train_df, validation_df, test_df = lemma_(train_df, validation_df, test_df)

# Encode Targets
train_labels, validation_labels, test_labels = target_encoder(train_df, validation_df)


def train_model(classifier, feature_vector_train, label, feature_vector_test, is_neural_net=False, type_="Word"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import plot_confusion_matrix
    from sklearn.metrics import f1_score, classification_report
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_test)

    plot_confusion_matrix(classifier, X=feature_vector_test, y_true=test_labels, normalize="true",
                          include_values=False)
    plt.title(type_)
    plt.grid(False)
    plt.savefig("Plots/ConfusionMatrices/" + type_ + "Captions" + str(len(predictions)) + ".png", bbox_inches='tight')
    plt.show()

    print(classification_report(test_labels, predictions, digits=3, zero_division=0))

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, test_labels)


def fit_classifiers():
    #### Feature Engineering #####
    # Create a Count Vectorizer object
    count_vect = CountVectorizer(analyzer='word', stop_words=stop_words)

    # The fit() method vectorises all of the strings in the data to give us a large vocabulary including all of the
    # words from all of the strings.
    # We only fit a vectoriser over the training data and not the validation data as if the word appears in the
    # validation data but not the training data then we can't learn from that word.
    count_vect.fit(train_df['Caption Lemma'])

    """
    The transform() method then cross references that vocabulary with each string and records a 1 if the word was 
    present and a 0 if not.
    xtrain_count is now a sparse array of length equal to the number of input examples and width equal to the total 
    count of all the words in all the input strings.
    This will be a sparse array as most elements will be zero as not all words appear in every string.
    """
    xtrain_count = count_vect.transform(train_df["Caption Lemma"])
    xtest_count = count_vect.transform(test_df["Caption Lemma"])
    # Naive Bayes on Count Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_labels, xtest_count, type_="Count Vectors")
    print("NB, Count Vectors: ", round(accuracy, 2))

    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=5000, max_df=.6)
    tfidf_vect.fit(train_df['Caption Lemma'])
    xtrain_tfidf = tfidf_vect.transform(train_df["Caption Lemma"])
    xtest_tfidf = tfidf_vect.transform(test_df["Caption Lemma"])
    # Naive Bayes on Word Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_labels, xtest_tfidf, type_="Word Level")
    print("NB, WordLevel TF-IDF: ", round(accuracy, 2))

    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), max_features=5000, stop_words="english",
                                       max_df=0.7)
    tfidf_vect_ngram.fit(train_df['Caption Lemma'])
    xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_df["Caption Lemma"])
    xtest_tfidf_ngram = tfidf_vect_ngram.transform(test_df["Caption Lemma"])
    # Naive Bayes on Ngram Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_labels, xtest_tfidf_ngram,
                           type_="Ngram")
    print("NB, N-Gram Vectors: ", round(accuracy, 2))

    # characters level tf-idf
    # We can't use stop_words on the character analyser, only word.
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=5000, max_df=0.7)
    tfidf_vect_ngram_chars.fit(train_df['Caption Lemma'])
    xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_df["Caption Lemma"])
    xtest_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(test_df["Caption Lemma"])
    # Naive Bayes on Character Level TF IDF Vectors
    accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_labels, xtest_tfidf_ngram_chars,
                           type_="Character Level")
    print("NB, CharLevel Vectors: ", round(accuracy, 2))


fit_classifiers()


# Not complete #
def voting_class(count_vect, tfidf_vect):
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import accuracy_score
    voting_types = ["soft", "hard"]
    for vt in voting_types:
        clf = VotingClassifier(estimators=[("cv", count_vect), ("tf", tfidf_vect)], voting=vt)
        clf.fit(train_df["Caption Lemma"], train_labels)
        y_pred = clf.predict(test_df["Caption Lemma"])
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(test_labels, predictions)
        print("Ensemble Classifier accuracy using", str(vt).capitalize(), "voting: %.2f%%" % (accuracy*100.0))

