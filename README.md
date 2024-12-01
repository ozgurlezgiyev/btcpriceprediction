
Bitcoin Price Prediction using SVR
This project demonstrates how to predict Bitcoin prices using Support Vector Regression (SVR) on historical price data. The project involves data preprocessing, training an SVR model, and visualizing the predicted Bitcoin prices compared to the actual historical prices.

Project Overview
This project uses historical Bitcoin price data to train an SVR model to predict future prices. The data is scaled using MinMaxScaler, and a non-linear RBF kernel is applied to the Support Vector Regression (SVR) model to capture complex patterns in the price movements.

The model predicts the Bitcoin prices for the next few days based on the historical price data of the last 60 days.

Key Steps:
Data Preparation: Historical Bitcoin price data is preprocessed and normalized.
Model Training: An SVR model is trained using the last 60 days of price data to predict the next day's price.
Prediction: Future Bitcoin prices are predicted, and the results are visualized.
Visualization: Both historical and predicted Bitcoin prices are plotted for comparison.
Installation
To run this project, make sure you have Python 3.x installed, and install the necessary dependencies using pip:

bash
pip install numpy pandas matplotlib scikit-learn
Usage
Prepare the Data:

Load your dataset containing Bitcoin prices (should have columns like Date and Close).
Ensure the Date column is in datetime format.
Run the Script:

Replace df with your DataFrame that contains the Bitcoin price data.
The script will preprocess the data, train the SVR model, and plot the historical vs predicted Bitcoin prices.
Code Execution:

Execute the script to train the model and generate the plot:
python
python bitcoin_price_prediction.py
Output:
The output will be a plot showing the actual Bitcoin prices and the predicted Bitcoin prices.
Example Output
Plot of Bitcoin Prices: The graph will show:
Historical Bitcoin Prices: Blue line.
Predicted Bitcoin Prices: Red line.
Example Code
python
Kodu kopyala
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

# Load data (ensure your DataFrame is named df)
# df = pd.read_csv('bitcoin_data.csv')

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Prepare training data for SVR
X_train, y_train = [], []
for i in range(60, len(scaled_data)):
    X_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Fit the SVR model
model_svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
model_svr.fit(X_train, y_train)

# Prepare test data
X_test = []
for i in range(60, len(scaled_data)):
    X_test.append(scaled_data[i-60:i, 0])
X_test = np.array(X_test)

# Predict the prices
predicted_scaled = model_svr.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))

# Plot predictions
predicted_index = df.index[60:]
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], label='Historical')
plt.plot(predicted_index, predicted_price, label='Predicted', color='red')
plt.title('Bitcoin Price Prediction using SVR')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

Requirements
Python 3.x

Libraries: numpy, pandas, matplotlib, scikit-learn

Author
Ozgur Lezgiyev

Feel free to modify the code for your specific use case. If you encounter any issues or have suggestions for improvements, feel free to open an issue or create a pull request.
