import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

import sys

sys.path.append(".")

from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation


def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, "data.pkl.gzip")

    f = gzip.open(data_file, "rb")
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype("float32")
    y = np.array(data["action"]).astype("float32")

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = (
        X[: int((1 - frac) * n_samples)],
        y[: int((1 - frac) * n_samples)],
    )
    X_valid, y_valid = (
        X[int((1 - frac) * n_samples) :],
        y[int((1 - frac) * n_samples) :],
    )
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space
    #    using action_to_id() from utils.py.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    X_train = np.array([rgb2gray(img) for img in X_train])
    X_valid = np.array([rgb2gray(img) for img in X_valid])

    y_train = np.array([action_to_id(a) for a in y_train])
    y_valid = np.array([action_to_id(a) for a in y_valid])

    def build_history(X, history_length):
        N = X.shape[0]
        X_hist = np.zeros((N, history_length, 96, 96), dtype=np.float32)
        for i in range(N):
            for h in range(history_length):
                idx = max(0, i - h)
                X_hist[i, history_length - 1 - h] = X[idx]
        return X_hist

    X_train = build_history(X_train, history_length)
    X_valid = build_history(X_valid, history_length)

    X_train = X_train / 255.0
    X_valid = X_valid / 255.0

    return X_train, y_train, X_valid, y_valid


def sample_minibatch(X, y, batch_size):
    class_counts = np.bincount(y, minlength=5).astype(np.float32)
    class_weights = 1.0 / np.sqrt(class_counts + 1e-6)
    sample_weights = class_weights[y]
    sample_weights /= sample_weights.sum()
    indices = np.random.choice(len(y), batch_size, replace=True, p=sample_weights)
    return X[indices], y[indices]


def compute_accuracy(agent, X, y):
    outputs = agent.predict(X)
    predictions = outputs.argmax(dim=1).cpu().numpy()
    return np.mean(predictions == y)


def train_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_minibatches,
    batch_size,
    lr,
    history_length=1,
    model_dir="./models",
    tensorboard_dir="./tensorboard",
):
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    # Instantiate agent
    # action_to_id maps to 5 classes: 0=straight, 1=left, 2=right, 3=accelerate, 4=brake
    n_classes = 5
    agent = BCAgent(history_length=history_length, n_classes=n_classes, lr=lr)

    # TODO: specify your agent with the neural network in agents/bc_agent.py
    # agent = BCAgent(...)
    tensorboard_eval = Evaluation(
        tensorboard_dir,
        "Imitation Learning",
        stats=["train_loss", "train_accuracy", "val_accuracy"],
    )
    # TODO: implement the training
    #
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    #
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     if i % 10 == 0:
    #         # compute training/ validation accuracy and write it to tensorboard
    #         tensorboard_eval.write_episode_data(...)

    # TODO: save your agent
    # model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    # print("Model saved in file: %s" % model_dir)

    # Training loop
    for i in range(n_minibatches):
        X_batch, y_batch = sample_minibatch(X_train, y_train, batch_size)
        loss = agent.update(X_batch, y_batch)

        if i % 10 == 0:
            train_acc = compute_accuracy(agent, X_batch, y_batch)
            valid_acc = compute_accuracy(agent, X_valid, y_valid)
            print(
                f"[{i}/{n_minibatches}]  loss: {loss:.4f}  "
                f"train_acc: {train_acc:.3f}  val_acc: {valid_acc:.3f}"
            )
            tensorboard_eval.write_episode_data(
                i,
                {
                    "train_loss": loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": valid_acc,
                },
            )

    # Save agent
    save_path = os.path.join(model_dir, f"bc_agent_history_length_{history_length}.pt")
    agent.save(save_path)
    print("Model saved in file: %s" % save_path)
    tensorboard_eval.close_session()


if __name__ == "__main__":
    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")

    history_length = 8

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid, history_length=history_length
    )

    # train model (you can change the parameters!)
    train_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        n_minibatches=3000,
        batch_size=256,
        lr=1e-4,
        history_length=history_length,
    )
