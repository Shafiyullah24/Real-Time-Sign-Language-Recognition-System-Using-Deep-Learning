import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


DATA_DIR = "dataset_custom"
IMG_SIZE = (28, 28)    # Match your inference code: cv2.resize(gray, (28, 28))
BATCH_SIZE = 32
EPOCHS = 20
VALIDATION_SPLIT = 0.2
SAVE_MODEL_FILE = "sign_language_model_custom.keras" # Matches your inference code
SAVE_INDICES_FILE = "class_indices.npy"             # Matches your inference code

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset directory '{DATA_DIR}' not found. Please create it with subfolders for each sign (e.g., A/, B/, C/).")

print("Loading images from directory...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale', # Use grayscale here!
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale', # Use grayscale here!
    label_mode='categorical'
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# Save the class indices mapping for the inference script
class_indices = {name: i for i, name in enumerate(class_names)}
np.save(SAVE_INDICES_FILE, class_indices)
print(f"Class indices saved to {SAVE_INDICES_FILE}")


# --- 2. Build the Model (A simple, common CNN structure) ---
def create_model(input_shape, num_classes):
    model = Sequential([
        # Rescale layer is crucial: Matches the /255.0 normalization in inference
        Rescaling(1./255, input_shape=input_shape),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Helps prevent overfitting
        Dense(num_classes, activation='softmax') # Softmax for multi-class classification
    ])
    return model

# Input shape is (height, width, channels). Channels is 1 for grayscale.
input_shape = IMG_SIZE + (1,)
model = create_model(input_shape, num_classes)

# --- 3. Compile and Train ---
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callback to save the best model weights
checkpoint = ModelCheckpoint(
    SAVE_MODEL_FILE, 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

print("\n--- Starting Training ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# --- 4. Evaluate (Optional, but recommended) ---
print("\n--- Final Evaluation ---")
# Load the best version of the model saved by the callback
best_model = tf.keras.models.load_model(SAVE_MODEL_FILE)

# Note: For proper testing, you should reserve a separate 'test' subset
# of data, but we'll use the validation set here for simplicity.
loss, accuracy = best_model.evaluate(val_ds)
print(f"Best Model Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

print(f"\nTraining complete. Model saved as: {SAVE_MODEL_FILE}")
print("You can now use this model file in your inference script.")