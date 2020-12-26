# Convolutional Neural Networks

# To run code in PyCharm console, highlight and press ⌥⇧E (option shift E)

# Data Preprocessing
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

tf.__version__

# Preprocess the training set
# We'll perform Image Augmentation: randomly transform some of the images to prevent overfitting.
# Just to add more variety to the training set
train_datagen = ImageDataGenerator(
	rescale=1. / 255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
	'DeepLearning/Convolutional Neural Networks (CNN)/dataset/training_set',
	target_size=(64, 64),
	batch_size=32,
	class_mode='binary'
)

# Preprocess the test set
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
	'DeepLearning/Convolutional Neural Networks (CNN)/dataset/test_set',
	target_size=(64, 64),
	batch_size=32,
	class_mode='binary'
)

# Initialize the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(
	filters=32,
	kernel_size=3,
	activation='relu',
	input_shape=[64, 64, 3]
))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Add a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(
	filters=32,
	kernel_size=3,
	activation='relu'
))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Now train the CNN
# Compile the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train on the training set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# Make a single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('DeepLearning/Convolutional Neural Networks (CNN)/dataset/prediction/cat.jpg',
	target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
print(training_set.class_indices)
if result[0][0] == 1:
	prediction = 'dog'
else:
	prediction = 'cat'

print(prediction)

