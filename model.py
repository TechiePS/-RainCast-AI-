import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# Load dataset
data_path = r'/content/combined_rainfall_data_2000_to_2023.csv'  # Update this path
rainfall_data = pd.read_csv(data_path)

# Check for NaN values
rainfall_data.dropna(inplace=True)

# Preprocessing function
def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Season'] = df['Month'].apply(lambda x: 'Winter' if x in [12, 1, 2] else
                                       'Spring' if x in [3, 4, 5] else
                                       'Summer' if x in [6, 7, 8] else
                                       'Fall')
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
    df.drop(columns='Date', inplace=True)
    scaler = MinMaxScaler()
    df['Rainfall (mm)'] = scaler.fit_transform(df[['Rainfall (mm)']])
    return df, scaler

# Preprocess the data
rainfall_data, scaler = preprocess_data(rainfall_data)

# Train model and save it
def train_and_save_model(city_name):
    city_data = rainfall_data[rainfall_data['City'].str.lower() == city_name.lower()].copy()

    if city_data.empty:
        raise ValueError(f"No data available for city: {city_name}")

    city_data.drop(columns='City', inplace=True)

    # Adding lag features
    city_data['Lag1'] = city_data['Rainfall (mm)'].shift(1)
    city_data['Lag2'] = city_data['Rainfall (mm)'].shift(2)
    city_data.dropna(inplace=True)

    X = city_data[['Year', 'Month', 'Day', 'Lag1', 'Lag2']]
    y = city_data['Rainfall (mm)']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Neural Network model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        Dropout(0.3),
        Dense(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

    # Save the model
    model_file = f'{city_name.lower()}_nn_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

# Load trained model
def load_model(city_name):
    model_file = f'{city_name.lower()}_nn_model.pkl'
    if not os.path.exists(model_file):
        raise ValueError(f"Model for city '{city_name}' does not exist. Please train the model first.")

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    return model

# Add this function to calculate percentage change
def calculate_percentage_change(previous, current):
    if previous == 0:
        return float('inf')
    return ((current - previous) / previous) * 100

def predict_rainfall(model, year, month, day, last_rainfall):
    input_data = np.array([[year, month, day, last_rainfall, last_rainfall]])
    prediction = model.predict(input_data).flatten()[0]
    predicted_rainfall = scaler.inverse_transform(np.array([[prediction]]))[0][0]

    # non-negative predicted rainfall
    predicted_rainfall = max(0, predicted_rainfall)

    # Calculate chance of rain
    chance_of_rain = min(100, max(0, predicted_rainfall * 10)) if predicted_rainfall > 0 else 0

    return predicted_rainfall, chance_of_rain

# Train the model for a specific city
def prepare_model(city_name):
    # Check if data exists for the city before training
    if city_name.lower() not in rainfall_data['City'].str.lower().unique():
        raise ValueError(f"No data available for city '{city_name}'. Please check the city name.")

    # If the model doesn't exist, train it
    if not os.path.exists(f'{city_name.lower()}_nn_model.pkl'):
        print(f"Training model for {city_name}...")
        train_and_save_model(city_name)
    else:
        print(f"Model for {city_name} already exists. Skipping training.")

# Main function to ask for city name and train the model
def main():
    # Ask for the city name
    city_name = input("Enter the city name: ").strip()

    # Prepare the model for the specified city
    prepare_model(city_name)
    print(f"Model for {city_name} is ready!")

# Run the main function
if __name__ == "__main__":
    main()
