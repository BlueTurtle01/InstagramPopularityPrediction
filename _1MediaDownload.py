import os
import pathlib
from ClassifierDetails import usernames_list


def download_media():
    """
    Download only the image and the relevant metadata about the image.
    :return: NA
    """
    path = pathlib.Path(__file__).parent  # Fetch the file path for this file.
    for user in usernames_list:
        # Only download new images that don't already exist in the folders.
        if os.path.isdir((str(path) + "/Input/" + user)):
            pass
        else:
            try:
                os.system("instagram-scraper " + str(user) + " --media-metadata --media-types image")
            except:
                pass
    print("All the media has been downloaded")


download_media()


