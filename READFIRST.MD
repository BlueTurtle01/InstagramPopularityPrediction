# Description of Pipeline

## Installation
The text based classifiers will run on a CPU but the CNN will take over 6 hours to run on the CPU
so it is advised to run this on a GPU. To do so you will first need to install the relevant packages.
I did this on an NVIDIA RTX 3090 and have included the package installation links below. They are very detailed so
I think repeating the information here is unnecessary.

https://www.tensorflow.org/install/gpu.
Specifically pay attention to the Software Requirements section.

You will need to create folders called Train, Test, Validation. The script will create the subdirectories of those for each Class.

Zip: https://drive.google.com/file/d/1IOr1PujzrxtPg8o1-Xdze6FMLL6aTd7Y/view?usp=sharing

Option 1 is much faster:
1. You will also need to download the zip file of the input images and create another folder Called Input, and unzip the contents to there.
   The <code>_4PreProcessing.py</code> file will then copy the respective files to their subdirectories as necessary.
2. Create the Input folder and run the <code>_1MediaDownload.py</code> file which will check if a user from the usernames_list exists in the Input folder, if not then
to download its contents. The folder for each user gets downloaded to the project root however, so these folders need to be then moved into the Input folder.

## Python Package Install
To install the relevant Python packages open a terminal, navigate to where the requirements.txt file is located
and enter <code>pip install -r requirements.txt</code> I suggest doing this in a fresh Conda environment so as to not affect the
package installs of your main Python installation that the Operating System will use.

## Description of Folder Layout
<code>Plots</code> contains the output images from the various functions.<br>
<code>Validation, Train, Test</code> contain subdirectories which are named after the relevant Class
that they contain.<br>
<code>Input</code> Contains the input images and JSON files before any pre-processing.
These are copied to the relevant folders after we have split the dataset into Train, Test, Validation, but a copy of
the originals is kept here. The <code>_1MediaDownload.py</code> file will check this folder
to see if a user already exists before downloading a second copy of the data. Therefore, it is advised to keep this folder.<br>

## Order of Execution
1. <code>_1MediaDownload.py</code> Downloads the user information and images.
2. <code>_2UsernameParser.py</code> Reads the JSON from step one and parses it to a .csv for later processing.
3. <code>_3CaptionCleaner.py</code> Cleans the captions, removes unicode characters, newlines, etc.
4. <code>_4Preprocessing.py</code> Conducts the train/validation/test split of the data, as well as moving the respective images
to their respective folder for train/test/validation, and then into the respective subfolder depending on the estimated Ratio
   calculated in <code>_2UsernameParser.py</code>. <code>EDA.py</code> Conducts Exploratory Data Analysis.
   
5. <code>BayesClassifierCaptions.py</code>, <code>BayesClassifierMentions.py</code>, and <code>BayesClassifierTags.py</code> 
   do as their name suggests and run Bayes classifiers for the text content. They take details from <code>ClassifiersDetails.py</code>
   and hence running that file alone does not do anything.
   
6. <code>CSVANN.py</code> Creates multiple tree methods; XGBoost, Decision Tree, Random Forest, and AdaBoost, on the meta-data
of the images that it reads from the CSV files. In addition to this we also create and train an Artificial Neural Network
   for the same input information.
   
7. <code>TextStackedModel.py</code> Conducts similar to the individual Bayes Classifier files, but the input is stacked first
before analysing.
   
8. <code>CNN.py</code> Creates a pre-trained model and adds some finals layers before training and validating on our data.
It then tests this model and output an accuracy score. This file saves a CSV for both the mode prediction and the argmax prediction to be used in
   <code>CombinedModels.py</code>.
   
9. <code>CombinedModels.py</code> takes the outputs of <code>CSVANN.py</code>, <code>CNN.py</code>, and <code>TextStackedModel.py</code>
to make an average and mode prediction for each image after analysing the text meta-data and the image content. This gives us a prediction
   based on the total Post content and is the main aim of the project.
   
<code>Plots.py</code> is a utility file and will be read at different points in the above 9 files, but does not output
anything when ran directly.

# Potential Issues
This was originally coded on MacOS, and then finished on Windows. It was at this switch over that I realised that the use of / versus \ between the
two operating systems causes problems in the regex in <code>CNN.py</code>. It is now coded for the Windows system. To run on MacOS change the line:
<code>names = [re.sub(r'^\d+\.\d+[\\\\]', '', file) for file in ds_test.filenames]</code> to 
<code>names = [re.sub(r'^\d+\.\d+[/]', '', file) for file in ds_test.filenames]</code>