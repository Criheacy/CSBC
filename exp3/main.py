import numpy as np
import matplotlib.pyplot as plt

config = {
    "image_width": 5,
    "image_height": 6
}


def visualize_data(data, title='', cmap='Greys'):
    plt.figure()
    plt.title(title)
    for _i in np.arange(10):
        plt.subplot(2, 5, _i + 1)
        plt.imshow(data[_i], cmap=cmap, interpolation='nearest')
        plt.xticks(np.arange(config["image_width"]))
        plt.yticks(np.arange(config["image_height"]))
    plt.tight_layout()
    plt.show()


def initialize():
    pixels = config["image_width"] * config["image_height"]
    weight = np.random.rand(pixels, pixels)
    return weight


def train(weight, train_data, iterations=100, learning_rate=0.001):
    loss_history = np.zeros(iterations)
    min_loss = np.inf
    best_weight = None
    for i in np.arange(iterations):
        data_flat = np.reshape(train_data, (10, -1))
        result = np.dot(data_flat, weight)

        weight -= np.dot(result.transpose(), data_flat) * learning_rate

        loss = np.average(result - data_flat)
        if loss < min_loss:
            min_loss = loss
            best_weight = weight
        loss_history[i] = loss
    return best_weight, loss_history


def predict(weight, predict_data):
    return np.dot(predict_data, weight)


# read data from file
data = np.zeros((10, config["image_height"], config["image_width"]), dtype=np.int)
for i in np.arange(10):
    f = open("res/%d.txt" % i)
    s = f.readlines()
    for line in np.arange(config["image_height"]):
        for char in np.arange(config["image_width"]):
            if s[line][char] == "#":
                data[i][line][char] = 1

# visualize_data(data)

noisy_data = np.copy(data)
for i in np.arange(10):
    for y in np.arange(config["image_height"]):
        for x in np.arange(config["image_width"]):
            if np.random.random_sample() > 0.9:
                noisy_data[i][y][x] = 1 - data[i][y][x]

data_diff = 0.3 * noisy_data + 0.7 * data

# visualize_data(data_diff)

w = initialize()

prediction_result = np.zeros((10, config["image_height"] * config["image_width"]))
for i in np.arange(10):
    prediction_result[i] = predict(w, np.reshape(noisy_data[i], -1))
prediction_result = np.reshape(prediction_result, (10, config["image_height"], config["image_width"]))
visualize_data(prediction_result)

train_iterations = 100
w, loss_history = train(w, data, train_iterations)

plt.figure()
plt.title("Loss History")
plt.xlim((0, 100))
plt.ylim((0, 7))
plt.plot(np.arange(train_iterations), loss_history)
plt.show()

prediction_result = np.zeros((10, config["image_height"] * config["image_width"]))
for i in np.arange(10):
    prediction_result[i] = predict(w, np.reshape(noisy_data[i], -1))
prediction_result = np.reshape(prediction_result, (10, config["image_height"], config["image_width"]))

visualize_data(prediction_result)
