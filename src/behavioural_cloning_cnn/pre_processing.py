import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy import ndimage
from sklearn.model_selection import train_test_split


class PreProcessingUtils:
    def __init__(self, csv_path: str):
        self.csv = csv_path
        self.csv_column_headings = [
            "center_path",
            "left_path",
            "right_path",
            "steering_angle",
            "throttle",
            "brake",
            "speed",
        ]
        self.df = self._create_dataframe()

    def _create_dataframe(self) -> pd.DataFrame:
        """
        :return: Take the instantiated CSV path and read to a dataframe with pre-specified column headings.
        """
        df = pd.read_csv(self.csv, engine="python")
        df.columns = self.csv_column_headings
        return df

    def add_correction_factor_columns(self, correction_factor: float) -> pd.DataFrame:
        """
        :param correction_factor: An arbitrary value to augment the steering angle based on camera pose.
        :return: This method takes our df created from our input CSV and augments it to add two steering angle columns
                 for the left and right cameras. Here we take an arbitrary value as we do not know the full pose of the
                 installed cameras.
        """
        self.df["steering_angle_left"] = self.df["steering_angle"] + correction_factor
        self.df["steering_angle_right"] = self.df["steering_angle"] - correction_factor

        return self.df.rename(
            columns={"steering_angle": "steering_angle_center"}, inplace=True
        )


class PreProcessData(PreProcessingUtils):
    def __init__(
        self,
        csv_path: str,
        correction_factor: float,
        process_center=True,
        process_left=False,
        process_right=False,
    ):
        """
        :param csv_path: The string path to the base csv.
        :param process_center: Bool to process center camera image.
        :param process_left: Bool to process left camera image.
        :param process_right: Bool to process right camera image.
        """
        super().__init__(csv_path)
        self.correction_factor = correction_factor
        self.center = process_center
        self.left = process_left
        self.right = process_right

    def plot_steering_distribution(self, title: str, show=False) -> plt.plot:
        """
        :param title: Set the title of the plot.
        :param show: If true display the plot else save the plot to disk.
        :return: A plot from the pandas dataframe of the distribution of the steering angles. NOTE: this expects a
                 column with the heading 'steering_angle'.
        """
        self.df["steering_angle"].hist(bins=40)
        plt.title(title)
        plt.xlabel("Steering Angle")
        plt.ylabel("Frequency")
        if show:
            plt.show()
        else:
            plt.savefig(f"./distribution_plots/{title}.jpg")

    def _return_columns_to_process(self) -> list:
        """
        :return: Return a list of the column headings to process based in instantiation parameters of the class.
        """
        columns = []
        if self.center:
            columns.append("center_path")
        if self.left:
            columns.append("left_path")
        if self.right:
            columns.append("right_path")

        return columns

    def lower_kurtosis(self) -> pd.DataFrame:
        """
        :return: Looks in the 'steering_angle' column and reduces the occurrence of the 0 steering angle by 30%.
        """
        self.plot_steering_distribution("before_kurtosis_lowering", show=False)
        self.df = self.df[self.df["steering_angle"] != 0].append(
            self.df[self.df["steering_angle"] == 0].sample(frac=0.7)
        )
        self.plot_steering_distribution("after_kurtosis_lowering", show=False)

        return self.df

    def create_datasets(self, test_set_size=0.2) -> np.array:
        """
        :param test_set_size: The percentage size of the test set from the total dataset.
        :return: Two numpy arrays containing the training and validation sets.
        """
        # Lower kurtosis of dataset
        self.lower_kurtosis()

        # Add correction factors
        self.add_correction_factor_columns(self.correction_factor)

        dataset = []
        for rows in self.df.itertuples():
            row = [
                rows.center_path,
                rows.left_path,
                rows.right_path,
                rows.steering_angle_center,
                rows.steering_angle_left,
                rows.steering_angle_right,
            ]
            dataset.append(row)

        # Shuffle the dataset
        dataset = np.asarray(dataset)
        dataset = sklearn.utils.shuffle(dataset)

        train_samples, validation_samples = train_test_split(
            dataset, test_size=test_set_size
        )

        return train_samples, validation_samples

    @staticmethod
    def run_generator(data_set, batch_size=32) -> np.array:
        """
        :param data_set: A numpy array containing the image_paths and their labels
        :param batch_size: The size of the batches we are going to feed to the model.
        :return: Yield a generator of the training set to conserve memory.
        """

        num_samples = len(data_set)

        while True:

            sklearn.utils.shuffle(data_set)

            for offset in range(0, num_samples, batch_size):

                batch_samples = data_set[offset : offset + batch_size]

                prefix = os.getcwd() + "/training_data/"

                images = []
                angles = []
                for batch_sample in batch_samples:
                    for i in range(0, 3):
                        try:
                            if i == 0:
                                img = ndimage.imread(prefix + batch_sample[0].strip())
                                images.append(img)
                                angles.append(batch_sample[3])
                                images.append(np.fliplr(img))
                                angles.append(-1.0 * float(batch_sample[3]))
                            elif i == 1:
                                img = ndimage.imread(prefix + batch_sample[2].strip())
                                images.append(img)
                                angles.append(batch_sample[4])
                                images.append(np.fliplr(img))
                                angles.append(-1.0 * float(batch_sample[4]))
                            elif i == 2:
                                img = ndimage.imread(prefix + batch_sample[2].strip())
                                images.append(img)
                                angles.append(batch_sample[5])
                                images.append(np.fliplr(img))
                                angles.append(-1.0 * float(batch_sample[5]))
                        except FileNotFoundError:
                            pass

                X = np.array(images)
                y = np.array(angles)

                yield sklearn.utils.shuffle(X, y)
