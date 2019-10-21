
import matplotlib.pyplot as plt
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical


# Import the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Prepare the image data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Prepare the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Network architecture
network = models.Sequential()
network.add(layers.Dense(units=512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(units=10, activation='softmax'))

# Compilation step
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Validation data set
img_val = train_images[:10000]
partial_img_train = train_images[10000:]

lbl_val = train_labels[:10000]
partial_lbl_train = train_labels[10000:]

# Train the network
history = network.fit(partial_img_train,
                      partial_lbl_train,
                      epochs=5,
                      batch_size=128,
                      validation_data=(img_val, lbl_val))

# Test the network
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('Test Accuracy:', test_acc)

# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# epochs = range(1, len(history_dict['accuracy']) + 1)
# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# plt.clf()
# acc_values = history_dict['accuracy']
# val_acc_values = history_dict['val_accuracy']
# plt.plot(epochs, acc_values, 'bo', label='Training Accuracy')
# plt.plot(epochs, val_acc_values, 'b', label='Validation Accuracy')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
