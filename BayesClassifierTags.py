import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
#My files
from ClassifierDetails import stop_words
from sklearn import  preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, f1_score, classification_report

train_df = pd.read_csv("CSVs/TrainingImages.csv")
test_df = pd.read_csv("CSVs/TestingImages.csv")

#Some images don't have hashtags so I need to replace the np.nan values with an empty string
train_df["Hashtags"] = train_df["Hashtags"].replace(np.nan, '', regex=True)
print("There are", len(train_df["Hashtags"]), "Training examples")
test_df["Hashtags"] = test_df["Hashtags"].replace(np.nan, '', regex=True)
print("There are", len(test_df["Hashtags"]), "Test examples")


#Encode the targets
encoder = preprocessing.LabelEncoder()
train_labels = encoder.fit_transform(train_df["Class"])
test_labels = encoder.fit_transform(test_df["Class"])

#### Feature Engineering ####
# create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words=stop_words)
count_vect.fit(train_df['Hashtags'])

# transform the training and test data using count vectorizer object
xtrain_count = count_vect.transform(train_df["Hashtags"])
xtest_count = count_vect.transform(test_df["Hashtags"])


tfidf_vect = TfidfVectorizer(analyzer='word', max_features=5000, stop_words=["the", "it", "of", "and", "a"])
tfidf_vect.fit(train_df['Hashtags'])
xtrain_tfidf = tfidf_vect.transform(train_df["Hashtags"])
xtest_tfidf = tfidf_vect.transform(test_df["Hashtags"])

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(train_df['Hashtags'])
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_df["Hashtags"])
xtest_tfidf_ngram = tfidf_vect_ngram.transform(test_df["Hashtags"])

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(train_df['Hashtags'])
xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_df["Hashtags"])
xtest_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(test_df["Hashtags"])


#### Create the Model ####
def train_model(classifier, feature_vector_train, label, feature_vector_test, is_neural_net=False, type_="Undefined"):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on test dataset
    predictions = classifier.predict(feature_vector_test)

    plot_confusion_matrix(classifier, X=feature_vector_test, y_true=test_labels, normalize="true",
                          include_values=False)
    plt.title(type_)
    plt.grid(False)
    plt.savefig("Plots/ConfusionMatrices/" + type_ + "Tags" + str(len(predictions)) + ".png", bbox_inches='tight')
    plt.show()

    #print(round(f1_score(test_labels, predictions, average='weighted',
    # labels=[int(x) for x in np.linspace(0, 14, 15)]), 2))

    print(classification_report(test_labels, predictions, digits=3, zero_division=0))

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, test_labels)


# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_labels, xtest_count, type_="Count Vectors")
print("NB, Count Vectors for tags:", round(accuracy, 2))

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_labels, xtest_tfidf, type_="Word Level")
print("NB, WordLevel TF-IDF for tags: ", round(accuracy, 2))


# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_labels, xtest_tfidf_ngram, type_="Ngram")
print("NB, N-Gram Vectors for tags: ", round(accuracy, 2))

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_labels, xtest_tfidf_ngram_chars, type_="Character Level")
print("NB, CharLevel Vectors for tags: ", round(accuracy, 2))

