import tensorflow as tf
from tensorflow.keras import layers
from include_file import load_data

if __name__ == '__main__':
    # Load Data
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # My Model
    model = tf.keras.models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
        layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        layers.Conv2D(256, (3, 3), activation='relu', padding="same"),
        layers.Conv2D(256, (3, 3), activation='relu', padding="same"),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        layers.Conv2D(512, (3, 3), activation='relu', padding="same"),
        layers.Conv2D(512, (3, 3), activation='relu', padding="same"),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        layers.Conv2D(512, (3, 3), activation='relu', padding="same"),
        layers.Conv2D(512, (3, 3), activation='relu', padding="same"),
        layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

    model.evaluate(x_test, y_test, verbose=2)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=64, epochs=10)

    print("now test")
    model.evaluate(x_test, y_test, verbose=2)
    print("complete")
