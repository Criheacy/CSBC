import numpy as np
import matplotlib.pyplot as plt
from neupy import algorithms

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


# read data from file
data = np.zeros((10, config["image_height"], config["image_width"]), dtype=np.int)
for i in np.arange(10):
    f = open("res/%d.txt" % i)
    s = f.readlines()
    for line in np.arange(config["image_height"]):
        for char in np.arange(config["image_width"]):
            if s[line][char] == "#":
                data[i][line][char] = 1

visualize_data(data)

noisy_data = np.copy(data)
for i in np.arange(10):
    for y in np.arange(config["image_height"]):
        for x in np.arange(config["image_width"]):
            if np.random.random_sample() > 0.9:
                noisy_data[i][y][x] = 1 - data[i][y][x]

data_diff = 0.3 * noisy_data + 0.7 * data

visualize_data(data_diff)

dhnet = algorithms.DiscreteHopfieldNetwork(mode='sync')
dhnet.train(data.reshape([10, -1])[5:, :])

# for number 5-9:
# dhnet.train(data.reshape([10, -1])[-5:, :])

predict_data = np.zeros_like(data)
for i in np.arange(10):
    predict_data[i] = dhnet.predict(noisy_data[i].flatten()) \
        .reshape([config["image_height"], config["image_width"]])

visualize_data(predict_data, cmap='Blues')
