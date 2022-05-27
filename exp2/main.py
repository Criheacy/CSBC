import numpy as np
import matplotlib.pyplot as plt

dataset = {
    "OR": {
        "x": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "y": np.array([0, 1, 1, 1])
    },
    "AND": {
        "x": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "y": np.array([0, 0, 0, 1])
    },
    "NOT": {
        "x": np.array([[0], [1]]),
        "y": np.array([1, 0])
    }
}

model = "OR"


def predicate(weight, bias, test_x):
    return np.dot(weight, test_x) + bias


def activate(predicate_y):
    # take the nearest y to simplify
    nearest_y = dataset[model]["y"][0]
    for i in np.arange(dataset[model]["y"].shape[0]):
        if np.abs(predicate_y - dataset[model]["y"][i]) < np.abs(predicate_y - nearest_y):
            nearest_y = dataset[model]["y"][i]
    return nearest_y


def get_loss(weight, bias):
    distance = 0
    for i in np.arange(dataset[model]["x"].shape[0]):
        distance += np.abs(predicate(weight, bias, dataset[model]["x"][i]) - dataset[model]["y"][i])
    return distance


def get_correctness(weight, bias):
    correct_count = 0
    for i in np.arange(dataset[model]["x"].shape[0]):
        if activate(predicate(weight, bias, dataset[model]["x"][i])) == dataset[model]["y"][i]:
            correct_count += 1
    return correct_count / dataset[model]["x"].shape[0]


def train(weight, bias, learning_rate=0.1):
    index = np.random.randint(dataset[model]["x"].shape[0])
    predicate_y = predicate(weight, bias, dataset[model]["x"][index])
    delta_y = predicate_y - dataset[model]["y"][index]
    return weight - learning_rate * np.dot(delta_y, dataset[model]["x"][index]), bias - learning_rate * delta_y


def init():
    # random in [-1, 1]
    weight = np.random.rand(*dataset[model]["x"][0, :].shape) * 2 - 1
    bias = np.random.rand() * 2 - 1
    return weight, bias


total_model = 10
train_round = 500

fig, loss_ax = plt.subplots()
plt.title(model + " model")

loss_ax.set_xlim(0, 300)
loss_ax.set_xlabel("training rounds")
loss_ax.set_ylabel("loss")
loss_ax.set_ylim(1, 5.5)
param_ax = loss_ax.twinx()
param_ax.set_ylim(-0.75, 1.25)
param_ax.set_ylabel("parameters")

for model_id in np.arange(total_model):
    w1_history = np.zeros(train_round)
    w2_history = np.zeros(train_round)
    b_history = np.zeros(train_round)
    loss_history = np.zeros(train_round)
    correctness_history = np.zeros(train_round)

    w, b = init()
    for i in np.arange(train_round):
        w1_history[i] = w[0]
        w2_history[i] = w[1]
        b_history[i] = b
        loss_history[i] = get_loss(w, b)
        correctness_history[i] = get_correctness(w, b)
        w, b = train(w, b)

    param_ax.plot(np.arange(train_round), w1_history, color="#f59211")
    param_ax.plot(np.arange(train_round), w2_history, color="#5cc0ff")
    param_ax.plot(np.arange(train_round), b_history, color="#d94fff")

    loss_ax.plot(np.arange(train_round), loss_history, color="#ff6666")

plt.legend(["w1", "w2", "b", "loss"], loc="upper right")
plt.tight_layout()
plt.show()
