import argparse

from behavioural_cloning_cnn.model import BehaviouralModel


def run():

    # Set configurable params
    parser = argparse.ArgumentParser(
        description="Run NN Training For Behavioural Cloning"
    )
    parser.add_argument(
        "--csv",
        nargs=1,
        type=str,
        default="./training_data/driving_log.csv",
        help="String path to CSV",
    )
    parser.add_argument(
        "--epochs",
        nargs=1,
        type=int,
        default=5,
        help="Integer number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        nargs=1,
        type=int,
        default=32,
        help="Size of batches to be processed",
    )
    parser.add_argument(
        "--keep_prob",
        nargs=1,
        type=float,
        default=0.2,
        help="The percentage of dropout in the network",
    )
    parser.add_argument(
        "--print_architecture",
        action="store_true",
        help="Print Model Architecture to STDOUT",
    )
    parser.add_argument(
        "--save_model_image",
        action="store_true",
        help="Same an image of the Model to disk",
    )
    args = parser.parse_args()

    # Run Training Pipeline

    model = BehaviouralModel(args.epochs, args.batch_size, args.keep_prob, args.csv[0])
    model.create_model(args.print_architecture, args.save_model_image)
    model.train_model()


__name__ == "__main__" and run()
