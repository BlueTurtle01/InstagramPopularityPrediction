import pandas as pd
import functools
import random
import numpy as np
from Plots import plot_conf_matrix
from Plots import plot3d_ensemble_accuracy
from sklearn.metrics import f1_score, classification_report


def combined_predictions_accuracy(results):
    # Calculate the accuracy of combining the ANN and Stacked Text Model
    matched = 0
    not_matched = 0

    for row in range(len(results)):
        if results.loc[row, "True Label"] == int(results.loc[row, "Ensemble prediction"]):
            matched += 1
        else:
            not_matched += 1

    #print("The accuracy of the Ensemble when using the Average prediction is:", round(matched / len(results), 1) * 100, "%")

    return round(matched / len(results), 2)


def combine_model_outcomes():
    """
    Each of our models has made a prediction of the Class each input belongs to.
    We will now take different weightings for the 3 models and calculate the overall prediction for the input.
    :return: NA
    """

    # First we read in the models
    TextStackedModel = pd.read_csv("CSVs/TextStackedModel.csv")
    CNN = pd.read_csv("CSVs/CNNPredictions.csv")
    CSV = pd.read_csv("CSVs/CSVANNPredictions.csv")

    # Create an empty list to store the tuples of weights (w1, w2, w3)
    weight_pairs = []

    # Merge the three dataframes together where they share the same Image_id
    Joineddf = functools.reduce(lambda left, right: pd.merge(left, right, on='Image_id'), [TextStackedModel, CNN, CSV])

    # Iterate over 200 different weightings
    for i in range(200):

        # Select 3 random proportions
        weight_nb = round(random.uniform(0, 1), 2)
        weight_csv = round(random.uniform(0, 1), 2)
        weight_cnn = round(random.uniform(0, 1), 2)

        # Normalise these random weights by the total sum so the weights all sum to 1.
        weight_nb = weight_nb / sum([weight_nb, weight_csv, weight_cnn])
        weight_cnn = weight_cnn / sum([weight_nb, weight_csv, weight_cnn])
        weight_csv = weight_csv / sum([weight_nb, weight_csv, weight_cnn])

        Joineddf["Ensemble prediction"] = round(weight_nb*Joineddf["NB Average Prediction"] + weight_cnn*Joineddf['CNN Prediction'] + weight_csv*Joineddf["CSV Prediction"], 0)

        accuracy = combined_predictions_accuracy(Joineddf)
        weight_pairs.append((weight_nb, weight_cnn, weight_csv, accuracy*100))

    # Sort all the different weightings by their respective accuracy scores.
    output = sorted(weight_pairs, key=lambda x: x[-1])

    # Select the weights that gave the best accuracy score
    best_weights = output[-1]

    weight_pairs = list(zip(*weight_pairs))
    plot3d_ensemble_accuracy(weight_pairs)

    # Rerun the ensemble model with the weights that gave the best accuracy score
    weight_nb, weight_csv, weight_cnn = best_weights[0], best_weights[1], best_weights[2]
    weight_nb = weight_nb / sum([weight_nb, weight_csv, weight_cnn])
    weight_cnn = weight_cnn / sum([weight_nb, weight_csv, weight_cnn])
    weight_csv = weight_csv / sum([weight_nb, weight_csv, weight_cnn])
    Joineddf["Ensemble prediction"] = round(
        weight_nb * Joineddf["NB Average Prediction"] + weight_cnn * Joineddf['CNN Prediction'] + weight_csv * Joineddf[
            "CSV Prediction"], 0)

    combined_predictions_accuracy(Joineddf)

    # Save the prediction of the most accurate ensemble model
    Joineddf.to_csv("CSVs/CombinedModel.csv", sep=",", index=False)

    f1 = round(f1_score(Joineddf["True Label"], Joineddf["Ensemble prediction"], average='weighted',
                        labels=[int(x) for x in np.linspace(0, 14, 15)]), 2)

    # Create a classification report for the ensemble model
    print(classification_report(Joineddf["True Label"], Joineddf["Ensemble prediction"], digits=3, zero_division=0))
    print("Accuracy:", best_weights[3], "%")

    plot_conf_matrix(Joineddf["True Label"], Joineddf["Ensemble prediction"], type_="Ensemble")


combine_model_outcomes()


def combine_probabilities():
    CSV = pd.read_csv("CSVs/CSVANNProbabilities.csv")
    CSV = CSV.sort_values(by="Image_id", ascending=True).set_index("Image_id")

    CNN = pd.read_csv("CSVs/CNNProbabilities.csv")
    CNN = CNN.sort_values(by="Image_id", ascending=True).set_index("Image_id")

    weight_pairs = []

    for i in range(200):
        weight_csv = round(random.uniform(0, 1), 2)
        weight_cnn = round(random.uniform(0, 1), 2)

        # Normalise these random weights by the total sum so the weights all sum to 1.
        weight_cnn = weight_cnn / sum([weight_csv, weight_cnn])
        weight_csv = weight_csv / sum([weight_csv, weight_cnn])

        CNN = CNN.mul(weight_cnn)
        CSV = CSV.mul(weight_csv)

        joined = pd.concat([CSV, CNN])
        joined = joined.groupby("Image_id").sum().reset_index()
        joined["Ensemble prediction"] = joined.apply(lambda row: np.argmax(row[1:22]), axis=1)

        accuracy = combined_predictions_accuracy(joined)
        weight_pairs.append((weight_cnn, weight_csv, accuracy*100))

    #plot3d_ensemble_accuracy(weight_pairs)

    plot_conf_matrix(joined["True Label"], joined["Ensemble prediction"], type_="Ensemble")

    joined.to_csv("CSVs/TestProbabilities.csv", index=False)
