import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#To see the distribution of ratios over all users I will plot a histogram. I expect the right tail to be very thin
#And the left tail to be much fatter.
def plot_ratios():
    training = pd.read_csv("CSVs/TrainingImages.csv")
    testing = pd.read_csv("CSVs/TestingImages.csv")
    validation = pd.read_csv("CSVs/ValidationImages.csv")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all', sharey="all")
    ax1.hist(training["Class"], density=True, bins=20)
    ax1.set_title("Distribution of class in the Training data")
    #ax1.xlabel("Class")
    #ax1.ylabel("Count per class")

    ax2.set_title("Distribution of class in the Testing data")
    #ax2.xlabel("Class")
    #ax2.ylabel("Count per class")
    ax2.hist(testing["Class"], density=True, bins=20)

    ax3.set_title("Distribution of class in the Validation data")
    #ax3.xlabel("Class")
    #ax3.ylabel("Count per class")
    ax3.hist(validation["Class"], density=True, bins=20)

    plt.show()


def plot_empirical(df):
    x, counts = np.unique(df["Class"], return_counts=True)
    cusum = np.cumsum(counts)
    y = cusum / cusum[-1]
    x = np.insert(x, 0, x[0])
    y = np.insert(y, 0, 0.)
    plt.plot(x, y, drawstyle='steps-post')
    plt.grid(True)
    plt.savefig('Plots/ecdf.png')


def plot_loss(history, type_):
    fig = plt.figure()
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Cross Entropy Loss')
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.legend()
    fig.savefig("Plots/" + type_ + "/Loss.png", bbox_inches='tight')


def plot_accuracy(history, type_):
    fig = plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    fig.savefig("Plots/" + type_ + "/Accuracy.png", bbox_inches='tight')


def plot_t(history):
    plt.title("Top 3/5 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")
    plt.plot(history.history['t3'], label='Top 3 Accuracy')
    plt.plot(history.history['t5'], label='Top 5 Accuracy')
    plt.legend()
    plt.show()


def plot_importances(classifier, Input, columns_list, clf_type):
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot the impurity-based feature importance of the classifier
    fig = plt.figure()
    plt.title(str("Feature Importances " + clf_type))
    plt.bar(range(Input.shape[1]), importances[indices], color="r", align="center")
    feature_names = [columns_list[i] for i in indices]  # Credit: https://stackoverflow.com/a/48023080/4367851
    plt.xticks(range(Input.shape[1]), feature_names, rotation=45)
    plt.xlim([-1, Input.shape[1]])
    plt.tight_layout()
    fig.savefig("Plots/" + str(clf_type) + ".jpg", bbox_inches='tight')


def plot_conf_matrix(Test_Labels, predictions, type_):
    from sklearn.metrics import confusion_matrix
    fig = plt.figure()
    cm = confusion_matrix(Test_Labels, predictions, normalize="true")
    plt.imshow(cm)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title('Confusion matrix')
    plt.show()
    fig.savefig("Plots/" + type_ + "/ConfusionMatrix.jpg", bbox_inches='tight')


def box_plot_class(df, y):
    # Credit: https://stackoverflow.com/a/49554843/4367851
    import seaborn as sns
    df["Class"] = round(df["Class"], 2)
    fig = plt.figure()
    sns.boxplot(x=df["Class"], y=df[y], whis=[5, 95], showfliers=False)
    plt.title((str(y) + " per Class"))
    plt.xlabel("Class")
    plt.xticks(rotation=90)
    plt.ylabel(y)
    fig.savefig("Plots/EDA/BoxPlot" + str(y) + ".jpg", bbox_inches='tight')
    plt.show()


def followers():
    data = pd.read_csv("CSVs/FollowerCount.csv")
    plt.hist(data["Follower Count"], bins=20)
    plt.show()


def time_bet_posts():
    """
    1. Plot the average number of posts per day against the number of followers the profile had.
    2. Calculate Pearson's correlation Coefficient
    :return:
    """
    import datetime
    from scipy.stats import pearsonr
    from numpy import inf

    data = pd.read_csv("CSVs/UsersImages.csv")
    post_count = data.groupby("Username")["Image_id"].count()
    post_count = pd.DataFrame(post_count).reset_index()
    temp_df = pd.DataFrame()
    for user in range(post_count.shape[0]):
        pc = post_count.loc[user, "Image_id"]

        username = post_count.loc[user, "Username"]
        user_data = data[data["Username"] == username]

        first_image = user_data["Date"].min()
        first_image = datetime.datetime.strptime(first_image, "%Y-%m-%d").date()

        last_image = user_data["Date"].max()
        last_image = datetime.datetime.strptime(last_image, "%Y-%m-%d").date()

        difference = abs((last_image - first_image).days)
        average_time = round(post_count.loc[user, "Image_id"] / difference, 2)

        temp_data = {"Username": username, "Post Count": pc, 'First_image': first_image, "Last_image": last_image,
                     "Posts_per_day": average_time}
        temp_df = temp_df.append(temp_data, ignore_index=True)

    follower_data = pd.read_csv("CSVs/FollowerCount.csv")
    new_df = pd.merge(follower_data, temp_df)
    new_df.to_csv("CSVs/FollowerCount2.csv")

    # We cant' calculate correlation if the series contain inf or nans.
    x = new_df["Posts_per_day"].fillna(0)
    x[x == inf] = 0

    y = new_df["Follower Count"].fillna(0)
    y[y == inf] = 0

    plt.scatter(x, y)
    plt.title("Number of posts per day versus Follower Count")
    plt.xlabel("Average number of posts per day")
    plt.ylabel("Follower Count")
    plt.savefig('Plots/EDA/AveragePosts.png')

    corr, _ = pearsonr(x, y)
    print('Pearson\'s correlation between number of posts per day and number of followers: %.3f' % corr)

    plt.show()


def count_posts():
    data = pd.read_csv("CSVs/UsersImages.csv")
    post_count = data.groupby("Username")["Image_id"].count().values
    plt.hist(post_count, bins=20)
    plt.show()


def corr_post_followers():
    """
    1. Plot the absolute number of posts against the number of followers the profile had.
    2. Calculate Pearson's correlation Coefficient
    :return:
    """
    from scipy.stats import pearsonr
    from numpy import inf
    # Find the username and post count
    post_data = pd.read_csv("CSVs/UsersImages.csv")
    post_data["Username"] = post_data["Username"].astype("string")
    post_count = post_data.groupby("Username")["Image_id"].count().reset_index()

    # Find the username and follower count
    follower_data = pd.read_csv("CSVs/FollowerCount.csv")

    # Merge the two dataframes together
    new_df = pd.merge(post_count, follower_data)
    new_df_ = new_df.rename(columns ={"Image_id" : "Post Count"}).set_index("Username")
    new_df_.to_csv("CSVs/FollowerCount.csv")

    # We cant' calculate correlation if the series contain inf or nans.
    x = new_df["Image_id"].fillna(0)
    x[x == inf] = 0

    y = new_df["Follower Count"].fillna(0)
    y[y == inf] = 0

    corr, _ = pearsonr(x, y)
    print('Pearson\'s correlation between number of posts and number of followers: %.3f' % corr)


    plt.scatter(x, y)
    plt.title("Absolute number of posts versus Follower Count")
    plt.xlabel("Number of Posts")
    plt.ylabel("Follower Count")
    # This is a nice graph but is distorted by those accounts that are older.
    # Consider using average followers per year.

    # Credit: https://www.kite.com/python/answers/how-to-plot-a-linear-regression-line-on-a-scatter-plot-in-python
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b)

    plt.savefig('Plots/EDA/AbsolutePosts.png')
    plt.show()


def plot_ensemble_accuracy(weight_pairs):
    fig = plt.figure()
    plt.scatter(weight_pairs[2], weight_pairs[0])
    plt.xlabel("Accuracy %")
    plt.ylabel("Weight for Naive Bayes")
    plt.title("Accuracy vs Ensemble component weighting")
    fig.savefig("Plots/CSVEnsembleAccuracy.jpg", bbox_inches='tight')
    plt.show()


def plot3d_ensemble_accuracy(weight_pairs):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)
    ax.plot_trisurf(weight_pairs[1], weight_pairs[0], weight_pairs[3], cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_xlabel("Weight of CNN")
    ax.set_ylabel("Weight of Naive Bayes")
    ax.set_zlabel("Accuracy %")
    ax.set_title("Ensemble Model Accuracy")
    fig.savefig("Plots/Ensemble/3dEnsembleAccuracy.jpg", bbox_inches='tight')
    plt.show()
