import pandas as pd
import re
from time import perf_counter

# Assumptions
"""
Unicode characters that are not punctuation marks make no substantial effect on the ratio.
    This is likely not true as people might be incentivised to click if they seem some smiling emojis, but this adds a lot of complexity to the Bayes
"""


def truncated(row):
    """
    :param row: Row for each image to analyse if the caption is over the Instagram 125 truncation limit
    :return: 1 if truncated, 0 otherwise
    """
    if row > 125:
        return 1
    else:
        return 0


def cap_split(data):
    try:
        # Remove all the unicode characters
        # cap = re.sub(r"\\u\S+", " ", cap) #This removes the unicode but these are often punctuation marks so I think I should keep these in.
        cap = re.sub(r'#\S+', "", data)
        cap = re.sub(r"@\S+", "", cap)  # Credit: https://www.machinelearningplus.com/python/python-regex-tutorial-examples/
        cap = re.sub(r"\\n\s+", "", cap)
        cap = re.sub(r"\\n", "", cap)
        cap = re.sub(r"#", "", cap)  # Remove singular hashtag characters

        dictionary = {"\\u2019": '\'', "\\ud83c": "", "\\ud83d": "", "\\udcaa": "", "\\udffc": "",
                      "\\u2022": "", "\\udffd": "", "\\ufe0f": "", "\\ud83e": "", "\\udf43": "",
                      }

        # Some unicode characters are not normal punctuation marks. I am unsure if I should include these as they have an
        # affect on the classification.
        """
        \ud83c runner Emoji
        \udf51 invalid character
        \ud83d smiley face with open mouth
        \udcaa invalid character
        \udffc invalid character
        \u2022 bullet point
        \udffd invalid character
        \ufe0f Cloud Emoji
        \ud83e Grinning Face
        \udf43 Leaf
    
        """

        for key in dictionary.keys():
            cap = cap.replace(key, dictionary[key])

        return cap

    except:
        return " "


def caption_cleaner():
    data = pd.read_csv("CSVs/UsersImages.csv")
    data["Caption Short"] = data.apply(lambda row: cap_split(row["Caption"]), axis=1)
    data["Caption Length"] = data.apply(lambda row: int(len(row["Caption Short"])), axis=1)
    data["CaptionTruncated"] = data.apply(lambda row: truncated(row["Caption Length"]), axis=1)
    data["Image_id"] = data.apply(lambda row: re.sub(r"\?tp=1&_nc_ht(.*)", "", row["Image_id"]), axis=1)

    data.to_csv("CSVs/UsersImages.csv", index=False)


# Using the lambda apply functions instead of for row in range... was 20x faster. I have deleted
# the old for loop code in favour of this.
tic = perf_counter()
caption_cleaner()
toc = perf_counter()
print(toc-tic)

