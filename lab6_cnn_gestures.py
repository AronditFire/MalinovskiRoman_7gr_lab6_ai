import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
DATASET_DIR = "dataset"
TEST_IMAGE_PATH = "test_gesture.jpg"

IMAGE_SIZE = (64, 64)

BATCH_SIZE = 32

EPOCHS = 5

USE_AUGMENTATION = False


def create_data_generators():
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(
            f"Не найдена папка с датасетом: {DATASET_DIR}. "
        )

    if not USE_AUGMENTATION:
        common_datagen = ImageDataGenerator( # настройка генератора
            rescale=1.0 / 255.0,
            validation_split=0.2,
        )

        train_data = common_datagen.flow_from_directory(
            DATASET_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="training",
            shuffle=True,
        )

        val_data = common_datagen.flow_from_directory(
            DATASET_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
        )

    # Если аугментация включена
    else:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
        )

        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            validation_split=0.2,
        )

        # Создаем обучающую выборку.
        train_data = train_datagen.flow_from_directory(
            DATASET_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="training",
            shuffle=True,
        )

        # Создаем валидационную выборку.
        val_data = val_datagen.flow_from_directory(
            DATASET_DIR,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
        )
    return train_data, val_data


def build_model(num_classes: int):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

            # Первый сверточный слой извлекает простые признаки
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),

            # Первый pooling
            tf.keras.layers.MaxPooling2D(2, 2),

            # Второй сверточный слой
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),

            # Второй pooling
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Flatten(),

            # Полносвязный слой
            tf.keras.layers.Dense(128, activation="relu"),

            # Выходной слой
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_training_plots(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Точность на обучении и валидации")
    plt.xlabel("Эпоха")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cnn_accuracy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Ошибка на обучении и валидации")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cnn_loss.png", dpi=150)
    plt.close()

def predict_single_image(model, class_names):
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"\nТестовое изображение не найдено: {TEST_IMAGE_PATH}")
        return

    img = load_img(TEST_IMAGE_PATH, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    predicted_probability = float(prediction[0][predicted_index])

    print("\nРезультат предсказания для test_gesture.jpg:")
    print("Предсказанный класс:", predicted_class)
    print("Уверенность модели:", round(predicted_probability * 100, 2), "%")


def main():
    train_data, val_data = create_data_generators()

    print("Найденные классы:", train_data.class_indices)
    class_names = list(train_data.class_indices.keys())

    model = build_model(num_classes=len(class_names))
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping( # добавил чтобы не было долгой обработки
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    # Обучаем модель на тренировочных данных и одновременно следим за качеством на валидации.
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[early_stopping],
    )

    save_training_plots(history)
    loss, accuracy = model.evaluate(val_data)
    print(f"\nТочность модели на валидации: {accuracy * 100:.2f}%")
    model.save("gesture_cnn_model.keras")
    print("Модель сохранена в файл gesture_cnn_model.keras")
    predict_single_image(model, class_names)


if __name__ == "__main__":
    main()
