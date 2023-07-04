import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l1_l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

# Here are some fake data for the sake of the example
# You should replace these with your real data
n_samples = 1000
n_features = 10
X = np.random.normal(size=(n_samples, n_features))
y = np.random.choice([0, 1], size=n_samples)

# Preprocessing: split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

def create_model(input_shape):
    model = Sequential()

    # First neural network
    model.add(Dense(128, activation='relu', input_shape=input_shape,
                    kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))  # L1 and L2 regularization
    model.add(Dropout(0.5))
    model.add(BatchNormalization())  # Batch normalization
    model.add(Dense(64, activation='relu',
                    kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))  # L1 and L2 regularization
    model.add(Dense(10, activation='relu'))

    # Second neural network
    model.add(Dense(128, activation='relu',
                    kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))  # L1 and L2 regularization
    model.add(Dropout(0.5))
    model.add(BatchNormalization())  # Batch normalization
    model.add(Dense(64, activation='relu',
                    kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))  # L1 and L2 regularization
    model.add(Dense(10, activation='relu'))

    # Third neural network
    model.add(Dense(128, activation='relu',
                    kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))  # L1 and L2 regularization
    model.add(Dropout(0.5))
    model.add(BatchNormalization())  # Batch normalization
    model.add(Dense(64, activation='relu',
                    kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))  # L1 and L2 regularization
    model.add(Dense(1, activation='sigmoid'))

    return model

model = create_model((n_features,))  # 10 features

# Define the optimizer
optimizer = Adam(learning_rate=0.001)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define the callbacks
callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
    EarlyStopping(monitor='val_loss', patience=10),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
]

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=callbacks)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")
