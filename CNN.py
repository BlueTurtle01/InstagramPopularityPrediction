# https://www.youtube.com/watch?v=q7ZuZ8ZOErE
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.framework import ops
from tensorflow.keras.models import load_model
ops.reset_default_graph()
from Plots import plot_conf_matrix, plot_loss, plot_accuracy


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

im_height = 250
im_width = 250
batch_size = 1024
test_batch_size = 256
seed = 123
epochs = 20

# create a data generator
train_datagen = ImageDataGenerator(rescale=1 / 255.)

test_datagen = ImageDataGenerator(rescale=1 / 255.)


ds_train = train_datagen.flow_from_directory(
    directory="Train/",
    color_mode="rgb",
    batch_size=batch_size,
    target_size=(im_height, im_width),
    shuffle=True,
    seed=seed,
    subset=None,
)

ds_validation = train_datagen.flow_from_directory(
    directory="Validation/",
    color_mode="rgb",
    batch_size=batch_size,
    target_size=(im_height, im_width),
    shuffle=True,
    seed=seed,
    subset=None,
)

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print(
        '\n\nThis error most likely means that this notebook is not '
        'configured to use a GPU.  Change this in Notebook Settings via the '
        'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
    raise SystemError('GPU device not found')


def create_model():
    with tf.device('/device:GPU:0'):
        model = ResNet50(include_top=False, weights='imagenet', input_shape=(im_width, im_height, 3))
        # Freeze the layers which you don't want to train. Here I am freezing the all layers.
        for layer in model.layers[:]:
            layer.trainable = False

        # Adding custom Layer
        x = model.output
        x = Flatten()(x)
        predictions = Dense(15, activation="softmax")(x)

        inputs = tf.keras.Input(shape=(im_width, im_height, 3))
        # creating the final model
        model_final = Model(model.inputs, outputs=predictions)

        # compile the model
        model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=0.0001, decay=0.6),
                            metrics=["accuracy"])

        # Save the model according to the conditions
        checkpoint = ModelCheckpoint("resnet50_Adam.h5", monitor='accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto', save_freq=epochs)
        early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')
        learning = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, verbose=1,
                                     mode='auto', min_delta=0.00001, cooldown=0, min_lr=0)

        # Train the model
        history = model_final.fit(
          ds_train,
          epochs=epochs,
          validation_data=ds_validation,
          callbacks=[checkpoint, early, learning])


#create_model()


def model_test(im_height, im_width, seed):

    import re
    model_final = load_model("resnet50_Adam.h5")
    # Create the test dataset

    ds_test = test_datagen.flow_from_directory(
        directory="Test/",
        color_mode="rgb",
        batch_size=test_batch_size,
        target_size=(im_height, im_width),
        shuffle=False,
        seed=seed)

    # Evaluate the model
    score = model_final.evaluate(ds_test)
    print(f'{score[1]:.2f}')

    # Credit: https://stackoverflow.com/a/53961683/4367851
    probs = model_final.predict(ds_test)

    # The DirectoryIterator saves the filenames including the directory name. We need to remove these.
    names = [re.sub(r'^\d+\.\d+[\\]', '', file) for file in ds_test.filenames]

    cols = list(range(1, 16, 1))
    cols.insert(0, "Image_id")
    cols.insert(len(cols), "True Label")
    probs_df = np.column_stack([np.array(names), probs, ds_test.labels])
    pd.DataFrame(probs_df, columns=[cols]).to_csv("CSVs/CNNProbabilities.csv", sep=',', index=False)

    y_pred = np.array([np.argmax(x) for x in probs])
    predictions_df = np.column_stack([np.array(names), y_pred])
    pd.DataFrame(predictions_df, columns=["Image_id", "CNN Prediction"]).to_csv("CSVs/CNNPredictions.csv", sep=",", index=False)


model_test(im_height, im_width, seed)
