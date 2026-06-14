from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

data = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = data.flow_from_directory(
    ".",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val = data.flow_from_directory(
    ".",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train, validation_data=val, epochs=5)

model.save("fruit_model.keras")
