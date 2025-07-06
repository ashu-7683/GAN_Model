import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Reshape, Conv2DTranspose, Conv2D, LeakyReLU, Flatten
from tensorflow.keras.models import Model, Sequential
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load COCO dataset (Placeholder, use a real dataset in practice)
text_descriptions = ["A red bird on a tree", "A car driving on a road", "A cat sitting on a sofa"]
image_data = np.random.randn(len(text_descriptions), 64, 64, 3)  # Fake image data (64x64 RGB images)

# Text Preprocessing
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(text_descriptions)
sequences = tokenizer.texts_to_sequences(text_descriptions)
padded_sequences = pad_sequences(sequences, maxlen=10)

# Define Generator
def build_generator():
    model = Sequential([
        Dense(256, activation='relu', input_shape=(10,)),  # Text input
        Reshape((4, 4, 16)),
        Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu'),
        Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu'),
        Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')  # 64x64 RGB Output
    ])
    return model

# Define Discriminator
def build_discriminator():
    model = Sequential([
        Conv2D(64, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(0.2), input_shape=(64, 64, 3)),
        Conv2D(128, kernel_size=4, strides=2, padding='same', activation=LeakyReLU(0.2)),
        Flatten(),
        Dense(1, activation='sigmoid')  # Real or Fake classification
    ])
    return model

# Build and compile models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False  # Freeze when training GAN

gan_input = tf.keras.Input(shape=(10,))
img = generator(gan_input)
validity = discriminator(img)
gan = Model(gan_input, validity)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training Function
def train_gan(epochs=1000, batch_size=2):
    for epoch in range(epochs):
        idx = np.random.randint(0, len(image_data), batch_size)
        real_imgs = image_data[idx]
        text_input = padded_sequences[idx]

        fake_imgs = generator.predict(text_input)

        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))

        g_loss = gan.train_on_batch(text_input, np.ones((batch_size, 1)))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss_real[0]} / {d_loss_fake[0]}, G Loss: {g_loss}")
            generate_sample(epoch)

# Generate Image Sample
 