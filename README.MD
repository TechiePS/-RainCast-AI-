 
 # RainCast AI 🌧️🤖
 
 ## 🌧️ Rainfall Prediction Project 🌧️

 ###  Overview

This project uses machine learning (specifically a neural network model) to predict the rainfall for a given city, year, month, and day based on historical weather data. The model is trained on a dataset spanning from 2000 to 2023 and predicts the amount of rainfall (in millimeters) and the chance of rain for a specific day.

----------------------------------------------------------------------------------------------------------------------------


 🚀 Features

- City-wise Rainfall Prediction: Predict rainfall for a specified city based on historical data.
- Chance of Rain: Calculate the chance of rain, based on the predicted rainfall.
- Interactive Web Interface: A Flask-based web interface where users can input city, year, month, and day to get predictions.

----------------------------------------------------------------------------------------------------------------------------

  ### 🛠️ Technologies Used

- Python: The main programming language used.
- TensorFlow & Keras: For building and training the neural network model.
- Pandas & NumPy: For data preprocessing and manipulation.
- Flask: To create the web application.
- Scikit-learn: For data splitting and scaling.
- Pickle: For saving and loading the trained models.

----------------------------------------------------------------------------------------------------------------------------

 ### 📊 Dataset

The dataset used for training contains historical rainfall data from 2000 to 2023. It includes the following columns:

- City: Name of the city.
- Date: Date and time of the recorded rainfall.
- Rainfall (mm): Amount of rainfall in millimeters.

The data is processed to extract additional features like Year, Month, Day, Season, and Scaled Rainfall, with the rainfall values scaled using MinMaxScaler.

----------------------------------------------------------------------------------------------------------------------------

 ### ⚙️ How to Run the Project

 1. Clone the Repository

git clone https://github.com/yourusername/rainfall-prediction-project.git
cd rainfall-prediction-project

2. Install Required Libraries
pip install -r requirements.txt

3. Prepare the Data and Train the Model
If you haven't trained the model yet, the first time you run the application, it will automatically train the model for the specified city. Ensure the dataset file (combined_rainfall_data_2000_to_2023.csv) is available in the directory.

4. Run the Flask App
To start the Flask server, run:
python app.py
The web app will be available at http://127.0.0.1:5000/.

5. Using the Web Interface
Enter the City Name.
Select the Year, Month, and Day for the prediction.
Click on Predict to get the predicted rainfall and chance of rain for that day.

----------------------------------------------------------------------------------------------------------------------------

 ### 🧑‍💻 Model Details
The model is a neural network built using Keras with the following layers:

Input layer with 5 features: Year, Month, Day, Lag1, Lag2.
Dense layers with ReLU activation, Batch Normalization, and Dropout for regularization.
Final output layer with a single neuron predicting rainfall (in mm).
🔄 Lag Features
The model uses Lag1 and Lag2 as additional features, which represent rainfall from the previous day and two days prior. This helps the model capture time-series patterns in the data.

🏋️‍♂️ Training
The model is trained for 150 epochs with a batch size of 32 using the Adam optimizer and Mean Squared Error as the loss function.

----------------------------------------------------------------------------------------------------------------------------

 ### ⚡ Example of How It Works
Input:
City: Mumbai
Year: 2023
Month: 8
Day: 15
Output:
Predicted Rainfall: 15.23 mm
Chance of Rain: 95.20%

----------------------------------------------------------------------------------------------------------------------------

### 🔄 Model Reload
The model is saved using Pickle to avoid retraining every time the app is run. If the model for a city doesn't exist, the application will automatically train and save the model.

----------------------------------------------------------------------------------------------------------------------------

### 📁 File Structure

.
├── app.py                 Main Flask application
├── model.py               Model training and prediction logic
├── combined_rainfall_data_2000_to_2023.csv   Raw data for training
├── requirements.txt       List of required libraries
├── rainfall_model.pkl     Saved model file (after training)
└── templates
    ├── index.html         Home page template
    └── results.html       Results page template


----------------------------------------------------------------------------------------------------------------------------


### OUTPUT
![Alt text](D:\project\PG\SEM 3\rainfall\RainfallPredictionProject-Neural Network\OUTPUT\1.PNG)
![Alt text](D:\project\PG\SEM 3\rainfall\RainfallPredictionProject-Neural Network\OUTPUT\2.PNG)
![Alt text](D:\project\PG\SEM 3\rainfall\RainfallPredictionProject-Neural Network\OUTPUT\3.PNG)


### ⚠️ Troubleshooting
Ensure the path to the dataset is correct (combined_rainfall_data_2000_to_2023.csv).
If you're using a new city, input the correct city name and wait for the model to train (this may take some time).
If the model is not found, the app will train it automatically, which may take time depending on the dataset size.
🤝 Contributing
Feel free to fork this repository, make changes, and submit pull requests. Contributions are always welcome! 😊

----------------------------------------------------------------------------------------------------------------------------

### 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

----------------------------------------------------------------------------------------------------------------------------

### 🎯 Future Improvements
Add more advanced features (e.g., weather-related features such as humidity, pressure).
Implement better model evaluation (e.g., cross-validation).
Use a more complex model, such as LSTM for time-series forecasting.

-----------------------------------------------------------------------------------------------------------------------------

### ✨ What's New:

- 🌧️ LSTM-based Model: Added a Long Short-Term Memory (LSTM) model for time-series forecasting of rainfall.
- 🤖 Combined Models: Introduced a combined approach that utilizes Neural Networks, XGBoost, CatBoost, and LSTM for improved prediction accuracy.


- 📁 New Model Files: 
 - rainfall_lstm_model.pkl: Saved model for the LSTM-based approach.
  - rainfall_nn_xgb_cb_lstm_model.pkl: Saved model for the combined Neural Network, XGBoost, CatBoost, and LSTM approach.


📊 Additional Data Files: 
  - rainfall_lstm_data.csv: Dataset specifically prepared for training the LSTM model.
  - rainfall_nn_xgb_cb_lstm_data.csv: Dataset for training the combined model (NN, XGBoost, CatBoost, and LSTM).

----------------------------------------------------------------------------------------------------------------------------
