
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical


# Import the MNIST dataset
(train_images, train_lables), (test_images, test_labels) = mnist.load_data()

# Prepare the image data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Prepare the labels
train_lables = to_categorical(train_lables)
test_labels = to_categorical(test_labels)

# Network architecture
network = models.Sequential()
network.add(layers.Dense(units=512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(units=10, activation='softmax'))

# Compilation step
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Train the network
network.fit(train_images, train_lables, epochs=5, batch_size=128)

# Test the network
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
