import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
# Load the dataset
df = pd.read_csv('ICICIBANK.csv')

# Handle missing values using linear interpolation
df.interpolate(inplace=True)

# Convert 'Date' column to datetime format and set it as the index
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.set_index('Date', inplace=True)

# Consider only 'Close' prices for prediction
data = df['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# Define the number of time steps
time_steps = 10

# Create sequences for training and testing sets
X, y = create_sequences(scaled_data, time_steps)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# Reshape the input data to be 3D (batch_size, time_steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Get the training loss history
train_loss = history.history['loss']

# Make predictions on the test set
predictions = model.predict(X_test)
predictions_original_scale = scaler.inverse_transform(predictions)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# Plot the training loss history
plt.plot(train_loss, label='Training Loss', color='green')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Plotting the actual vs predicted values
actual_values_original_scale = scaler.inverse_transform(y_test.reshape(-1, 1))
plt.plot(actual_values_original_scale, label='Actual', color='blue')
plt.plot(predictions_original_scale, label='Predicted', color='red')
plt.title('Reliance Stock Prices - Actual vs Predicted')
plt.xlabel('Time (oldest -> latest)')
plt.ylabel('Stock Closing Price')
plt.legend()
plt.show()