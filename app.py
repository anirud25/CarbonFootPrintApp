from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the pre-trained model
#pip3 install catboost
import pandas as pd
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
# # from sklearn.svm import SVR
# # from xgboost import XGBRegressor
# # from catboost import CatBoostRegressor
# # import lightgbm as lgb
# # from sklearn.linear_model import ElasticNet
# # from sklearn.tree import DecisionTreeRegressor

# Define the S3 bucket and file name
bucket_name = 'mydevawsbucket-23023'
file_name = 'lifestyle_sustainability_data.csv'

# Load the dataset
dataset = pd.read_csv(file_name)
print(dataset.shape)

X = dataset.drop(columns=['ParticipantID', 'CarbonFootprint', 'Recommendations'])
y = dataset['CarbonFootprint']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to train and evaluate a model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# Models with hyperparameters tuning
models = {
    # 'LinearRegression': LinearRegression(),
    # 'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    # 'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=200,learning_rate=0.1, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
    # 'CatBoost': CatBoostRegressor(iterations=100, depth=6, verbose=0, random_seed=42),
    # 'LightGBM': lgb.LGBMRegressor(n_estimators=200, num_leaves=31, random_state=42),
    # 'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    # 'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.7, random_state=42),
    # 'AdaBoost': AdaBoostRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    # 'DecisionTree': DecisionTreeRegressor(max_depth=4, random_state=42)
}

# Evaluate all models
for name, model in models.items():
    mse, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)
    print(f'{name} - Mean Squared Error: {mse:.4f}, R² Score: {r2:.4f}')


# # DL Models

# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense, Dropout
# # from tensorflow.keras.optimizers import Adam
# # import tensorflow as tf
# '''import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import warnings
# warnings.filterwarnings('ignore')

# X = dataset.drop(columns=['ParticipantID', 'CarbonFootprint', 'Recommendations'])
# y = dataset['CarbonFootprint']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Standardize the dataset for DL models
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Function to build and evaluate a model
# def evaluate_dl_model(model, X_train, X_test, y_train, y_test, epochs=100, batch_size=32):
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
#     model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     return mse, r2

# # 1. Deep Neural Network (DNN) Model
# def build_dnn_model(input_shape):
#     model = Sequential()
#     model.add(Dense(128, input_dim=input_shape, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(1))  # Output layer
#     return model

# # 2. Convolutional Neural Network (CNN) Model (1D for tabular data)
# def build_cnn_model(input_shape):
#     model = Sequential()
#     model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(input_shape, 1)))
#     model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
#     model.add(tf.keras.layers.Flatten())
#     model.add(Dense(50, activation='relu'))
#     model.add(Dense(1))  # Output layer
#     return model

# # 3. Long Short-Term Memory (LSTM) Model (1D for sequence/tabular data)
# def build_lstm_model(input_shape):
#     model = Sequential()
#     model.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(input_shape, 1)))  # Increased units
#     model.add(tf.keras.layers.Dropout(0.2))  # Added dropout for regularization
#     model.add(tf.keras.layers.LSTM(64))
#     model.add(Dense(32, activation='relu'))  # Added a dense layer
#     model.add(Dense(1))  # Output layer
#     return model

# # Evaluate DNN
# dnn_model = build_dnn_model(X_train.shape[1])
# mse_dnn, r2_dnn = evaluate_dl_model(dnn_model, X_train, X_test, y_train, y_test)
# print(f'DNN Model - Mean Squared Error: {mse_dnn:.4f}, R² Score: {r2_dnn:.4f}')

# # Reshape input data for CNN and LSTM models
# X_train_cnn_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test_cnn_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# # Evaluate CNN
# cnn_model = build_cnn_model(X_train.shape[1])
# mse_cnn, r2_cnn = evaluate_dl_model(cnn_model, X_train_cnn_lstm, X_test_cnn_lstm, y_train, y_test)
# print(f'CNN Model - Mean Squared Error: {mse_cnn:.4f}, R² Score: {r2_cnn:.4f}')

# # Evaluate LSTM
# lstm_model = build_lstm_model(X_train.shape[1])
# mse_lstm, r2_lstm = evaluate_dl_model(lstm_model, X_train_cnn_lstm, X_test_cnn_lstm, y_train, y_test, epochs=200, batch_size=64)
# print(f'LSTM Model - Mean Squared Error: {mse_lstm:.4f}, R² Score: {r2_lstm:.4f}')'''


# #GridSearchCV for feature importance

# from sklearn.model_selection import GridSearchCV

# # Define the parameter grid for hyperparameter tuning
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, 30, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }

# # Initialize the Random Forest model
# rf_model = RandomForestRegressor(random_state=42)

# # GridSearchCV to find the best hyperparameters
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)

# # Best parameters from grid search
# best_params = grid_search.best_params_

# # Train the model with the best parameters
# best_rf_model = RandomForestRegressor(**best_params, random_state=42)
# best_rf_model.fit(X_train, y_train)

# # Predicting Carbon Footprint with the refined Random Forest model
# y_best_rf_pred = best_rf_model.predict(X_test)

# # Model Evaluation
# best_rf_mse = mean_squared_error(y_test, y_best_rf_pred)
# best_rf_r2 = r2_score(y_test, y_best_rf_pred)

# best_params, best_rf_mse, best_rf_r2

# # Get feature importances
# feature_importances = best_rf_model.feature_importances_

# # Create a DataFrame to hold the feature importance scores
# importance_df = pd.DataFrame({
#     'Feature': X_train.columns,
#     'Importance': feature_importances
# }).sort_values(by='Importance', ascending=False)

# # Display the feature importance scores
# print(importance_df)

# #Best Model

# Select only the top 4 features based on feature importance
# top_features = ['EnergySource', 'DietType', 'TransportationMode', 'MonthlyElectricityConsumption']
# X = dataset[top_features]  # Top 4 features
# y = dataset['CarbonFootprint']  # Target variable

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Initialize the Gradient Boosting Regressor
# gbr_model = GradientBoostingRegressor(n_estimators=200, random_state=42)

# # Train the model
# gbr_model.fit(X_train, y_train)

# # Predict on the test set
# y_pred_gbr = gbr_model.predict(X_test)

# # Evaluate the model
# mse_gbr = mean_squared_error(y_test, y_pred_gbr)
# r2_gbr = r2_score(y_test, y_pred_gbr)

# # Print evaluation results
# print(f'Gradient Boosting Regressor (Top 4 Features) - Mean Squared Error: {mse_gbr:.4f}')
# print(f'Gradient Boosting Regressor (Top 4 Features) - R² Score: {r2_gbr:.4f}')

import random

def generate_recommendations(row):
    recommendations = []

    # EnergySource
    if row.get('EnergySource', None) == 2:  # Non-Renewable
        energy_source_options = [
            "Switch to renewable energy sources like solar or wind to reduce emissions.",
            "Consider installing solar panels or opting for green energy providers.",
            "Shifting to renewable energy can significantly cut down on your carbon footprint."
        ]
        recommendations.append(random.choice(energy_source_options))

    # TransportationMode
    if row.get('TransportationMode', None) == 2:  # Car
        transportation_options = [
            "Use public transit, biking, or walking to reduce transportation emissions.",
            "Carpool with others to minimize your carbon footprint.",
            "Try switching to eco-friendly transportation options like cycling or public transit."
        ]
        recommendations.append(random.choice(transportation_options))

    # DietType
    if row.get('DietType', None) == 2:  # Mostly Animal-Based
        diet_options = [
            "Shift towards a plant-based diet to significantly reduce diet-related emissions.",
            "Try incorporating more plant-based meals into your diet to lower your carbon footprint.",
            "Reducing animal-based products in your diet can make a big impact on the environment."
        ]
        recommendations.append(random.choice(diet_options))

    # UsingPlasticProducts
    if row.get('PlasticUsage', None) == 2:  # Often
        plastic_options = [
            "Cut down on single-use plastics and switch to sustainable alternatives.",
            "Invest in reusable products to limit plastic waste.",
            "Reducing plastic usage helps minimize harmful environmental impacts."
        ]
        recommendations.append(random.choice(plastic_options))

    # DisposalMethods
    if row.get('DisposalMethod', None) == 2:  # Landfill
        disposal_options = [
            "Adopt recycling and composting practices to manage waste effectively.",
            "Avoid landfill disposal by recycling and composting.",
            "Switch to eco-friendly disposal methods like composting or recycling."
        ]
        recommendations.append(random.choice(disposal_options))

    return "; ".join(recommendations)

    
import os
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, jsonify, send_from_directory

def plot_viz(df):
    energy_source_mapping = {0: 'Renewable', 1: 'Mixed', 2: 'Non-Renewable'}
    diet_type_mapping = {0: 'Plant-Based', 1: 'Balanced', 2: 'Animal-Based'}
    transport_mode_mapping = {0: 'Bike', 1: 'Public Transit', 2: 'Car', 3: 'Walk'}
    
    df = dataset.drop(columns=['ParticipantID', 'Recommendations'])
    
    # Create a directory to save the plots if it doesn't exist
    output_dir = 'static/plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.savefig(f'{output_dir}/correlation_heatmap.png')  # Save plot
    plt.close()
    
    # Pair plot for features
    sns.pairplot(df, hue='EnergySource', vars=['CarbonFootprint', 'MonthlyElectricityConsumption', 'DietType', 'TransportationMode'])
    plt.savefig(f'{output_dir}/pairplot.png')  # Save pairplot
    plt.close()

    df['EnergySource'] = df['EnergySource'].replace(energy_source_mapping)
    df['DietType'] = df['DietType'].replace(diet_type_mapping)
    df['TransportationMode'] = df['TransportationMode'].replace(transport_mode_mapping)
    
    # Example: Bar plot of EnergySource
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='EnergySource')
    plt.title('Distribution of Energy Source')
    plt.savefig(f'{output_dir}/energy_source_distribution.png')  # Save plot
    plt.close()

    # Example: Scatter plot of Monthly Electricity Consumption vs. Carbon Footprint
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='MonthlyElectricityConsumption', y='CarbonFootprint', hue='EnergySource')
    plt.title('Electricity Consumption vs Carbon Footprint')
    plt.savefig(f'{output_dir}/electricity_vs_carbon.png')  # Save plot
    plt.close()

    # Box plot of DietType vs Carbon Footprint
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='DietType', y='CarbonFootprint')
    plt.title('Carbon Footprint by Diet Type')
    plt.savefig(f'{output_dir}/diet_vs_carbon.png')  # Save plot
    plt.close()

    # Violin plot for Transportation Mode and Carbon Footprint
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x='TransportationMode', y='CarbonFootprint')
    plt.title('Carbon Footprint by Transportation Mode')
    plt.savefig(f'{output_dir}/transportation_vs_carbon.png')  # Save plot
    plt.close()


#plot_viz(dataset)

import boto3
import pickle
from flask import render_template

# Path where the model will be saved locally
model_filename = 'carbon_footprint_gbmodel.pkl'

# Save the trained model to a file
# with open(model_filename, 'wb') as file:
#     pickle.dump(gbr_model, file)

model = pickle.load(open('carbon_footprint_gbmodel.pkl', 'rb'))

app = Flask(__name__, template_folder='webapp',static_folder='webapp')
CORS(app) 

@app.route('/')
def home():
    return render_template("index.html")
    #return "<h1>Carbon Footprint Prediction Service is Active</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([data['EnergySource'], data['DietType'], data['TransportationMode'], data['MonthlyElectricityConsumption']])
    prediction = model.predict([features])[0]
    return jsonify({'prediction': prediction})

@app.route('/recommendations', methods=['POST'])
def recommend():
    data = request.get_json(force=True)
    print(data)
    recommendations = generate_recommendations(data)
    print(recommendations)
    return jsonify({'recommendations': recommendations})
   
@app.route('/static/plots/<path:filename>')
def download_file(filename):
    return send_from_directory('static/plots', filename)
 
@app.route('/plots')
def generate_plots():
    plot_viz(dataset)  # Assuming df is your dataset loaded in the app
    return jsonify({
        'message': 'Plots generated successfully!',
        'plot_urls': [
            '/static/plots/energy_source_distribution.png',
            '/static/plots/electricity_vs_carbon.png',
            '/static/plots/diet_vs_carbon.png',
            '/static/plots/transportation_vs_carbon.png',
            '/static/plots/correlation_heatmap.png',
            '/static/plots/pairplot.png'
        ]
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
