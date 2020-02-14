from math import ceil

from behavioural_cloning_cnn.pre_processing import PreProcessData
from keras.layers import (Activation, Cropping2D, Dense, Dropout, Flatten,
                          Lambda)
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils.vis_utils import plot_model


class BehaviouralModel:
    def __init__(self, epochs: int, batch_size: int, keep_prob: float, csv_path: str):
        """
        :param epochs: Integer of the number of epochs to train on.
        :param batch_size: Integer batch size per iteration.
        :param keep_prob: Float percentage of the dropout e.g. 0.2 will cause a dropout of 20%.
        :param csv_path: String path to the CSV containing locations of images and values.
        """
        # Define HyperParams
        self.EPOCHS = epochs
        self.KEEP_PROB = keep_prob
        self.BATCH_SIZE = batch_size

        # Define Data
        self.CSV_PATH = csv_path
        self._CORRECTION_FACTOR = 0.2

        self.pre_processor = PreProcessData(
            self.CSV_PATH,
            self._CORRECTION_FACTOR,
            process_center=True,
            process_left=True,
            process_right=True,
        )

        self.TEST_SET_PERCENT = 0.2
        self.training_set, self.validation_set = self.pre_processor.create_datasets(
            self.TEST_SET_PERCENT
        )
        self.X_train = self.pre_processor.run_generator(
            self.training_set, self.BATCH_SIZE
        )
        self.X_valid = self.pre_processor.run_generator(
            self.validation_set, self.BATCH_SIZE
        )

        # Instantiate Model
        self.model = Sequential()

    def create_model(self, print_architecture=False, plot_model_img=False):
        """
        Model based upon NVIDIA's DAVE-2 System:
        https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

        This is an inline model therefore it makes sense to use Keras sequential toolset.

        This model is a regression analysis to minimise the mean squared error of the output label to the model.
        The reasoning behind this is that we are trying to perform immitation learning to match that of what the human
        driver did therefore there is no need for a classification set.

        We can imagine this by visualising a graph with a regression line where the x-axis is the pose of the vehicle
        in relation to the center line of the lane and the y-axis is the steering input required to keep near the
        center.

        :param print_architecture: Bool, If True, prints the output of keras.model.summary() to STDOUT. Default False.
        :param plot_model_img: Bool, If True, saves a graphical image of the model to the cwd. Default False.
        :return: The nn model.
        """

        # Normalisation of the input image(s)
        self.model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

        # Cropping of the images to ROI, 50 pixels from top, 20 from bottom, Output=90x320x3
        self.model.add(
            Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3))
        )

        # Conv1
        self.model.add(
            Conv2D(filters=24, kernel_size=5, strides=(2, 2), padding="valid")
        )

        # Activation1
        self.model.add(Activation("relu"))

        # Conv2
        self.model.add(
            Conv2D(filters=36, kernel_size=5, strides=(2, 2), padding="valid")
        )

        # Activation2
        self.model.add(Activation("relu"))

        # Conv3
        self.model.add(
            Conv2D(filters=48, kernel_size=5, strides=(2, 2), padding="valid")
        )  # TODO: Validate number of filters vs image size

        # Activation3
        self.model.add(Activation("relu"))

        # Conv4 Input
        self.model.add(
            Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="valid")
        )

        # Activation4
        self.model.add(Activation("relu"))

        # Conv5 Input
        self.model.add(
            Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding="valid")
        )

        # Activation5
        self.model.add(Activation("relu"))

        # Add Dropout to Prevent Overfit
        self.model.add(Dropout(self.KEEP_PROB))

        # Flatten
        self.model.add(Flatten())

        # Fully Connected 1
        self.model.add(Dense(100))

        # Activation FC 1
        self.model.add(Activation("relu"))

        # Full Connected 2
        self.model.add(Dense(50))

        # Activation FC 2
        self.model.add(Activation("relu"))

        # Fully Connected 3
        self.model.add(Dense(10))

        # Activation FC 3
        self.model.add(Activation("relu"))

        # Output Layer
        self.model.add(Dense(1))

        if print_architecture:
            print(self.model.summary())

        if plot_model_img:
            plot_model(
                self.model,
                to_file="model_plot.png",
                show_shapes=True,
                show_layer_names=True,
            )

        return self.model

    def train_model(self):
        """
        :param use_tensorboard: Bool, if True, saves log data for use with Tensorboard. Default, False.
        :return: A trained model.
        """

        # Compile model using the Adam Optimiser and Reducing Loss Through Mean Squared Error
        self.model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])

        # Train data using a generator
        self.model.fit_generator(
            generator=self.X_train,
            epochs=self.EPOCHS,
            verbose=1,
            steps_per_epoch=ceil(len(self.training_set) / self.BATCH_SIZE),
            validation_data=self.X_valid,
            validation_steps=ceil(len(self.validation_set) / self.BATCH_SIZE),
        )

        self.model.save("model.h5")
