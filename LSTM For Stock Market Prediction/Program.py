from flask import Flask, render_template, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_graph')  
def generate_graph():
    def generate_stock_prediction_graph():
        try:
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

            # Define and compile the model
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                LSTM(units=50, return_sequences=True),
                LSTM(units=50),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            history = model.fit(X_train, y_train, epochs=51, batch_size=32, verbose=0)

            # Get the training loss history
            train_loss = history.history['loss']

            # Make predictions on the test set
            predictions = model.predict(X_test)
            predictions_original_scale = scaler.inverse_transform(predictions)

            # Plotting the actual vs predicted values
            actual_values_original_scale = scaler.inverse_transform(y_test.reshape(-1, 1))
            plt.plot(actual_values_original_scale, label='Actual', color='red')
            plt.plot(predictions_original_scale, label='Predicted', color='green')
            plt.title('Stock Prices - Actual vs Predicted')
            plt.xlabel('Time (oldest -> latest)')
            plt.ylabel('Stock Closing Price')
            plt.legend()

            # Convert the plot to bytes for returning as graph data
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            return buffer

        except Exception as e:
            print("Error:", e)
            return None

    # Example usage:
    graph_data = generate_stock_prediction_graph()
    if graph_data:
        return send_file(graph_data, mimetype='image/png')
    else:
        return "Error generating graph"

if __name__ == '__main__':
    app.run(debug=True)
