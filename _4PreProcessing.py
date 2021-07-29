import pandas as pd
import shutil
import os
import pathlib
import numpy as np
from time import perf_counter
from sklearn.model_selection import train_test_split


path_ = pathlib.Path(__file__).parent  # Fetch the file path for this file.


def split_dataset():
    data = pd.read_csv("CSVs/UsersImages.csv")

    train, test = train_test_split(data, test_size=0.1)
    print("There are", train.shape[0], "total images")
    print("Number of Testing images:", test.shape[0])
    train, validation = train_test_split(train, test_size=0.25)
    print("Number of Training images:", train.shape[0])
    print("Number of Validation images:", validation.shape[0])

    test.to_csv("CSVs/TestingImages.csv", sep=",", index=False)
    validation.to_csv("CSVs/ValidationImages.csv", sep=",", index=False)
    train.to_csv("CSVs/TrainingImages.csv", sep=",", index=False)

    return train, test, validation


split_dataset()


def folder_delete():
    """
    Every time we run the split_dataset function it will create a new train-test-split.
    If this is immediately followed by the move_func it will move some files in addition to those already in the
    Train, Test, Validation folders. If we repeat this many times we will eventually end up with all of the files in
    all of the folders, and the network will be training, validating, and testing on the same images.

    To remove this possibility, every time we run the split_dataset function we will delete all of the sub directories
    inside Train, Test, Validation first.
    """
    from glob import glob

    for subdir in glob('Train/*'):
        shutil.rmtree(str(subdir))

    for subdir in glob("Test/*"):
        shutil.rmtree(str(subdir))

    for subdir in glob("Validation/*"):
        shutil.rmtree(str(subdir))


# Create folders for the input images to be moved to if the folder doesn't already exist.
def folder_creator():
    folder_delete()
    folders = [round(float(x), 3) for x in np.linspace(0.0, 0.35, 15)]
    types = ["/Train/", "/Test/", "/Validation/"]
    for type_ in types:
        for i in range(len(folders)):
            try:
                os.mkdir("".join([str(path_), str(type_), str(folders[i])]), mode=0o777)
            except FileNotFoundError:
                pass


def move_func(file_name, user, ratio, type_):
    """
    :param file_name: string, filename which is <image_id>.jpg
    :param user: string, username
    :param ratio: ratio of likes/followers
    :param type_: string, Train, Validation, Test.
    :return: NA
    """
    ext = os.path.splitext(file_name)[-1].lower()
    ratio = round((0.025 * round(ratio / 0.025)), 3)
    if ext == ".jpg":
        if 0 <= ratio <= 0.35:
            current_dir = "".join([str(path_), "/Input/", str(user), "/", str(file_name)])
            new_dir = "".join([str(path_), str(type_), str(ratio), "/", str(file_name)])
            if os.path.isfile(new_dir):
                pass
            else:
                try:
                    shutil.copy(current_dir, new_dir)
                except FileNotFoundError:
                    pass
        else:
            pass
    else:
        pass


folder_creator()


def media_divider(type_):
    """
    For each username folder we want to:
    1. Enter the folder
    2. Pick the first image
    3. Search the UserImages.csv file for that image name
    4. Parse the Ratio value from that row
    5. Classify depending on ratio
    6. Move image to the relevant folder.

    :param type_: string, Train, Test, Validation
    :return: NA
    """
    if type_ == "/Test/":
        data = pd.read_csv("CSVs/TestingImages.csv")
        data.apply(lambda row: move_func(row["Image_id"], row["Username"], row["Class"], type_), axis=1)
    elif type_ == "/Train/":
        data = pd.read_csv("CSVs/TrainingImages.csv")
        data.apply(lambda row: move_func(row["Image_id"], row["Username"], row["Class"], type_), axis=1)
    else:
        data = pd.read_csv("CSVs/ValidationImages.csv")
        data.apply(lambda row: move_func(row["Image_id"], row["Username"], row["Class"], type_), axis=1)


tic = perf_counter()
media_divider("/Test/")
media_divider("/Train/")
media_divider("/Validation/")
toc = perf_counter()
print(toc - tic)
