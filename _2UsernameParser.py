import json
import re  # Used for regular expressions in the image_id and username to remove quotation marks.
from datetime import datetime  # media date-time is recording using a timestamp which needs to be converted.
from ClassifierDetails import usernames_list
from instaloader import Instaloader, Profile
import pandas as pd
import numpy as np
import time
from datetime import timedelta
import pathlib
import cv2
import math
from datetime import date
from time import perf_counter


def log_followers(x1, x2, y1, y2, x_new):
    """
    Estimate the number of followers the profile had when the image in question was posted.
    :param x1: Number of days elapsed when account was created - this is technically 0 but causes a DivisonByZero error
    so I chose to use 1. I left this as a variable though for future changes.
    :param x2: Int value of days account since creation
    :param y1: Number of followers at creation: 0
    :param y2: Number of followers today
    :param x_new: Int value of days between image posting and x2
    :return: Estimated number of followers at a particular date, or 0.
    """
    # We know that the logarithm function is y = a + b*ln(x) -> followers = a + b *ln(age of account)
    try:
        a = y1 - (y2-y1)/math.log(x2/x1)*math.log(x1)
        b = (y2-y1)/math.log(x2/x1)
        y_new = a + b * math.log(x_new)
        return y_new
    except ValueError:
        return 0


def calc_sat(image_id, username):
    # Credit: https://stackoverflow.com/a/58831691/4367851
    """
    Calculate the Saturation of the image
    :param image_id: String id to locate the image in the Input directory.
    :param username: String type for username
    :return: The hue of the image, or 0 if unable to calculate for the image.
    """
    try:
        image_id, _ = image_id.split("?tp", 2)
        image_path = "".join(["Input/", str(username), "/", str(image_id)])
        img = cv2.imread(image_path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        sat = round(img_hsv[:, :, 1].mean(), 2)
        return sat
    except:
        return 0


def calc_hue(image_id, username):
    """
    Calculate the Hue of the image
    :param image_id: String id to locate the image in the Input directory.
    :param username: String type for username
    :return: The hue of the image, or 0 if unable to calculate for the image.
    """
    # Credit: https://stackoverflow.com/a/58831691/4367851
    # Image Hue
    try:
        image_id, _ = image_id.split("?tp", 2)
        image_path = "".join(["Input/", str(username), "/", str(image_id)])
        img = cv2.imread(image_path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue = round(img_hsv[:, :, 0].mean(), 2)
        return hue
    except:
        return 0


def age_of_image(Date):
    """
    Calculates the age of each image in number of days since today.
    :param Date: A date object for a particular image
    :return: Age of the image in days, otherwise 0.
    """
    imageage = date.today() - Date   # Calculate the number of days the account had been open when the image was posted
    try:
        imageage = str(imageage).split(",")[0].split(" ")[0]  # Split away the word "days"
        return int(imageage)
    except:
        return 0


def difference_between_posts(df):
    """
    Calculate the number of days between the current post and the previous post.
    :param df: Dataframe of the particular user's media-metadata
    :return: Integer days between two posts, or 0 for the first image.
    """
    df["DaysSinceLastPosting"] = (df["Date"].diff(periods=1) / np.timedelta64(1, 'D'))  # Credit: https://stackoverflow.com/a/18215499/4367851
    df["DaysSinceLastPosting"] = df["DaysSinceLastPosting"].fillna(0)  # The most current image has an empty cell, fill this with zero.
    return df


def retrieve_follower_count(username):
    # Credit: https://stackoverflow.com/a/52984722/4367851
    """
    This will allow us to retrieve the follower count for any public profile.
    Steps:
    1. Scrape the list of usernames for the image to a separate file, likely .csv
    2. Loop through the .csv username file  with the below function and save the outcome as a .csv
    3. Join this .csv with the one created that has (usernames, image_id)
    """
    tic = time.perf_counter()
    # Instantiate the loader
    L = Instaloader()

    # Retrieve username content
    profile = Profile.from_username(L.context, username)

    # Append the username and follower count for this user to the temporary list.
    temp_list.append({"Username": username, "Follower Count": profile.followers})
    toc = time.perf_counter()
    print(f"Downloaded {username} in {toc - tic:0.4f} seconds")

    # Create a dataframe from the list of username/follower count data.
    df = pd.DataFrame(temp_list)

    # Save dataframe as a .csv
    df.to_csv("CSVs/FollowerCount.csv", index=False)


def collect_media_single(username):
    """
    Collect all of the meta-data for all of the images for one single user together in a dataframe
    :param username: The username of the current user were are parsing
    :return: A df for this particular user
    """
    path = pathlib.Path(__file__).parent  # Fetch the file path for this file.
    json_file_path = "".join([str(path), "/Input/", str(username), "/", str(username), ".json"])

    # Load in the json data
    with open(json_file_path, encoding="utf-8") as f:
        data = json.load(f)

        # Credit: https://stackoverflow.com/a/15816174/4367851
        num_images = len([i for i in data["GraphImages"] if i['username'] == str(username)])

    # Parse follower count
    follower_df = pd.read_csv("CSVs/FollowerCount.csv")
    follower_count = follower_df[follower_df["Username"] == username]
    follower_count = follower_count["Follower Count"]

    image_list = []
    for image in range(num_images):
        try:
            # Parse the number of likes
            count = json.dumps(data["GraphImages"][image]["edge_media_preview_like"]["count"], indent = 4)

            # Parse image_id that corresponds with the file names of the images
            image_id = json.dumps(data["GraphImages"][image]["thumbnail_resources"][0]["src"], indent=4).strip('\"')
            image_id = re.split("\?_nc_ht", image_id)

            # The image meta-data is the same for all size images, the actual image is currently in it's original size,
            # and not 150x150.
            image_id = re.split("s150x150/", str(image_id[0]))

            caption = json.dumps(data["GraphImages"][image]["edge_media_to_caption"]["edges"][0]["node"]["text"],
                                 indent=4).strip('\"')

            # The time and date may affect the popularity of a photo.
            timestamp = json.dumps(data["GraphImages"][image]["taken_at_timestamp"], indent=4)
            date_time = datetime.fromtimestamp(int(timestamp))
            date = datetime.strptime(str(date_time), '%Y-%m-%d %H:%M:%S')
            # This is the date object when the image was posted.
            # date.time() gives us the time and date.date() gives us the date.

            # Parse the username.
            username = json.dumps(data["GraphImages"][image]["username"], indent=4).strip('\"')
            image_list.append({"Image_id": image_id[1], "Username": username, "Likes": count, "Date": date.date(),
                               "Time": date.time(), "Caption": caption})
        except IndexError:
            pass

    df = pd.DataFrame(image_list)
    # Credit: https://stackoverflow.com/questions/6331497/an-elegant-way-to-get-hashtags-out-of-a-string-in-python
    df["Hashtags"] = df.apply(lambda row: [words.strip("#") for words in row["Caption"].split() if words.startswith("#")], axis=1)
    df["Mentions"] = df.apply(lambda row: [words.strip("@") for words in row["Caption"].split() if words.startswith("@")], axis=1)

    df["Mention Count"] = df.apply(lambda x: len(x["Mentions"]), axis=1)
    df["Hashtag Count"] = df.apply(lambda x: len(x["Hashtags"]), axis=1)
    df["Image Age"] = df.apply(lambda row: age_of_image(row["Date"]), axis=1)

    age_of_account = int((date.today().date() - df["Date"].min()) / timedelta(days=1))
    df["Image Age From Start"] = df.apply(lambda x: int((x["Date"] - df["Date"].min())/timedelta(days=1)), axis=1)
    df["Log Followers"] = df.apply(lambda row: log_followers(1, age_of_account, 0, follower_count,
                                                             row["Image Age From Start"]), axis=1)

    # If the number of images has been limited then this variable represents the time since the first image in the period, not the account open date.
    # I believe Follower growth is logarithmic but I will leave this here in case in future this assumption changes
    # followers_per_day = follower_count / int((date.today().date() - df["Date"].min()) / timedelta(days=1))
    # df["Estimated Linear Followers"] = df.apply(lambda row: round(int(follower_count) - row["Image Age"] * float(followers_per_day), 2), axis=1)
    # df["Ratio Linear"] = df.apply(lambda row: round(int(row["Likes"]) / (row["Estimated Linear Followers"] + 0.001), 2), axis=1)
    # df["Class Linear"] = df["Ratio Linear"].apply(lambda x: 0.025 * round(x / 0.025)) # Credit: https://stackoverflow.com/a/2272174/4367851

    df["Ratio Log"] = df.apply(lambda row: round(int(row["Likes"]) / (row["Log Followers"] + 0.001), 2), axis=1)
    df["Class"] = df["Ratio Log"].apply(lambda x: 0.025 * round(x / 0.025))
    # Credit: https://stackoverflow.com/a/2272174/4367851

    df["Hue"] = df.apply(lambda row: calc_hue(row["Image_id"], row["Username"]), axis=1)
    df["Saturation"] = df.apply(lambda row: calc_sat(row["Image_id"], row["Username"]), axis=1)

    df = difference_between_posts(df)

    df = df[df["Class"] < 0.36]
    df = df[df["Saturation"] != 0]
    df = df[df["Hue"] != 0]

    return df


tic = perf_counter()

# Create a temporary list to hold the iterated username/count data until it is stored in the final dataframe.
temp_list = []


def collect_media_total():
    """
    Iterates over all users in the usernames_list and conducts:
    1. Retrieve the current follower count
    2. Collect all of the media-metadata for the particular user from the JSON file downloaded alongside the images.
    3. Appends this user's information to the main list
    :return: NA
    """
    results = []

    # For each user in the username list we run the function to collect the metadata about the media from the JSON file.
    for i in usernames_list:
        try:
            retrieve_follower_count(i)
            data = collect_media_single(i)
            results.append(data)
        except:
            pass

    df = pd.concat(results, axis=0)

    df.to_csv("CSVs/UsersImages.csv", sep=",", index=False)


collect_media_total()

toc = perf_counter()
print(toc - tic)