import tensorflow as tf
import numpy as np
import time
import platform


def build_model(input_shape, classes=10):
    inputs = tf.keras.layers.Input(input_shape)

    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model_ = tf.keras.models.Model(inputs, x)
    return model_


def build_vgg(input_shape, classes=10):
    inputs = tf.keras.layers.Input(input_shape)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, padding='same')(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, padding='same')(x)

    x = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, padding='same')(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model_ = tf.keras.models.Model(inputs, x)
    return model_


class PerformanceLogger(tf.keras.callbacks.Callback):
    """Benchmarking training performance
        Logging time to train a model, time per epoch, and hardware information.
    """

    def __init__(self, filepath=None):
        self.train_start_time = None
        self.epoch_start_time = None

        self.epoch_elapsed_time = []
        self.train_elapsed_time = None

        self.filepath = filepath

    # training start
    # fit() start
    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()

    # epoch start
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    # epoch end
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_elapsed_time.append(time.time() - self.epoch_start_time)

    # training end
    def on_train_end(self, logs=None):
        self.train_elapsed_time = time.time() - self.train_start_time

        # system information
        system_info = platform.system()
        system_name = "system: {}".format(system_info)
        platform_info = "platform: {}".format(platform.platform())
        tf_version = "tf.__version__: {}".format(tf.__version__)
        gpu_list = "tf.config.list_physical_devices(): {}".format(tf.config.list_physical_devices('GPU'))

        # only mac
        if system_info == "Darwin":
            system_name = "{} ({})".format(system_name, platform.machine())

        # result
        train_time = "training: {}s ({} epochs)".format(int(self.train_elapsed_time), len(self.epoch_elapsed_time))
        epoch_time = "epoch: {}s".format(
            int(np.mean(self.epoch_elapsed_time)) if np.mean(self.epoch_elapsed_time) >= 1 else round(
                np.mean(self.epoch_elapsed_time), 3))

        # print result
        result = [system_name, platform_info, tf_version, gpu_list, train_time, epoch_time]

        for result_ in result:
            print(result_)

        if self.filepath is not None:
            with open(self.filepath, "w") as f:
                for result_ in result:
                    print(result_, file=f)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


if __name__ == "__main__":
    print(tf.__version__)
    print(tf.config.list_physical_devices('GPU'))

    # Load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # Add channel-axis
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(x_train.shape)
    print(x_test.shape)

    # Preprocessing
    batch_size = 128
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(normalize_img).cache().shuffle(
        len(x_train)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(normalize_img).batch(
        batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)

    # Build model
    model = build_model(input_shape=x_train.shape[1:])
    # model = build_vgg(input_shape=x_train.shape[1:])

    # Compile model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"]
    )

    # Train model
    model.fit(train_ds, epochs=10, validation_data=test_ds, callbacks=[PerformanceLogger()])
