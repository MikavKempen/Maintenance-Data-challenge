# Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import random
from itertools import product

from Task_a import dict_to_features_labels  # Import your existing preprocessing functions
from Task_a import data_train_norm, data_valid_norm, data_test_norm  # Or load saved normalized data

## FUNCTIONS
def build_ann_model(input_dim, hidden_units=[64, 32], dropout_rate=0.2, lr=0.001):

    model = Sequential()
    model.add(Dense(hidden_units[0], activation='relu', input_shape=(input_dim,)))
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

def build_and_train_ann(X_train, y_train, X_valid, y_valid,
                        hidden_units=None, dropout_rate=None, lr=None,
                        epochs=50, batch_size=32, verbose=1, plot_loss=True):
    ### Building the model
    input_dim = X_train.shape[1]
    model = build_ann_model(input_dim, hidden_units=hidden_units, dropout_rate=dropout_rate, lr=lr)

    ### Training the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    if plot_loss:
        plot_training_history(history)

    return model, history

def hyperparameter_tuning_ann(X_train, y_train, X_valid, y_valid,
                              hidden_units_list=[[128, 64], [64, 32]],
                              dropout_rates=[0.2], learning_rates=[0.001],epochs=50,
                              batch_size=32, verbose=0, plot_best=True):

    best_val_acc = -np.inf
    best_model = None
    best_history = None
    best_params = None

    # Iterate over all hyperparameter combinations
    for hu, dr, lr in product(hidden_units_list, dropout_rates, learning_rates):
        model, history = build_and_train_ann(X_train, y_train, X_valid, y_valid,
                                             hidden_units=hu,
                                             dropout_rate=dr,
                                             lr=lr,
                                             epochs=epochs,
                                             batch_size=batch_size,
                                             verbose=verbose,
                                             plot_loss=False)

        val_acc = max(history.history['val_accuracy'])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            best_history = history
            best_params = {'hidden_units': hu, 'dropout_rate': dr, 'lr': lr}

    if plot_best:
        print(f"Best Hyperparameters: {best_params}, Validation Accuracy: {best_val_acc:.4f}")
        plot_training_history(best_history)

    return best_model, best_params, best_history

## EXECUTION

### Lock seeds
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

### Convert normalized dictionaries to feature matrices
X_train, y_train, _ = dict_to_features_labels(data_train_norm)
X_valid, y_valid, _ = dict_to_features_labels(data_valid_norm)
X_test, y_test, _   = dict_to_features_labels(data_test_norm)

### Build and train model
model, history = build_and_train_ann(X_train, y_train, X_valid, y_valid,
                                     hidden_units=[128, 64], dropout_rate=0.2, lr=0.001,
                                     epochs=50, batch_size=32)

### Tune hyperparameters of model
best_model, best_params, best_history = hyperparameter_tuning_ann(X_train, y_train, X_valid, y_valid,
    hidden_units_list=[[128, 64], [64, 32]], dropout_rates=[0.1, 0.2],
    learning_rates=[0.001, 0.0005], epochs=5, batch_size=32)

