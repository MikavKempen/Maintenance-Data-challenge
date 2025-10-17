# Imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History
from Task_a import dict_to_features_labels  # Import your existing preprocessing functions
from Task_a import data_train_norm, data_valid_norm, data_test_norm  # Or load saved normalized data

## FUNCTIONS
def build_ann_model(input_dim, hidden_units=[128, 64], dropout_rate=0.2, lr=0.001):

    model = Sequential()
    model.add(Dense(hidden_units[0], input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout_rate))
    for units in hidden_units[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history, save_path=None):

    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

## EXECUTION

### Convert normalized dictionaries to feature matrices
X_train, y_train, _ = dict_to_features_labels(data_train_norm)
X_valid, y_valid, _ = dict_to_features_labels(data_valid_norm)
X_test, y_test, _   = dict_to_features_labels(data_test_norm)

### Build model
input_dim = X_train.shape[1]
model = build_ann_model(input_dim)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Plot training/validation loss
plot_training_history(history)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
