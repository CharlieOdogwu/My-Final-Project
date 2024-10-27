
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("OneDrive/Documents/CASMIR'S DOCUMENT/Superstore_sales_prediction_cleaned")

# Assuming 'Diagnosis' column contains the labels for whether a person has breast cancer or not
X = df.drop(columns=['Sales'])  # Features
y = df['Sales']  # Target labels

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a Sequential model for regression
model = Sequential()

# Use Input layer to define input shape
model.add(Dense(16, activation='relu', input_shape=(len(X.columns),)))  # Define input shape explicitly

# Add Dense layers with adjustments
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())  # Optional: add batch normalization for stable learning
model.add(Dropout(0.2))  # Lower dropout rate
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))

# Output layer for regression (no activation or linear activation)
model.add(Dense(1, activation='linear'))  # Linear activation for regression

# Compile the model with regression-appropriate loss and metrics
model.compile(optimizer=Adam(learning_rate=0.0015), 
              loss='mse',  # Mean Squared Error for regression
              metrics=['mae'])  # Mean Absolute Error as a metric
# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss',  # Monitor validation loss
                               patience=5,          # Stop after 5 epochs with no improvement
                               restore_best_weights=True)  # Restore the best model weights

# Compile the model for regression (use MSE or MAE for loss)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Fit the model with EarlyStopping
history = model.fit(X_train_scaled, y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_split=0.15, 
                    verbose=1, 
                    callbacks=[early_stopping])  # Add the EarlyStopping callback

# Evaluate the model's performance on the test data using Mean Absolute Error (MAE)
mae = model.evaluate(X_test_scaled, y_test)
print(f'Mean Absolute Error on Test Set: {mae}')

# Define the predict_sales function
def predict_sales(input_data):
    return model.predict(input_data)
              
# Streamlit App
logo_path = r"OneDrive/Documents/CASMIR'S DOCUMENT/My Comany.jpg"
st.image(logo_path, use_column_width='auto')

# Streamlit app title
st.title("Eso's & Grandsons Superstore Prediction")

# App description
st.write("""
Predict future sales based on input features like Category, Sub-Category, Region, Segment, Shipmode, Delivery time
""")

# Sidebar for user inputs
st.header('Input Parameters')

def user_input_features():
    category = st.selectbox("Category", ["Furniture", "Office Supplies", "Technology"])
    region = st.selectbox("Region", ["East", "West", "Central", "South"])
    sub_category = st.selectbox("Sub-Category", ["Accessories", "Appliances", "Art", "Blinders", "Bookcases", "Chairs", "Copiers", "Envelopes", "Fasteners", "Furnishings", "Labels", "Machines", "Paper", "Phones", "Storage", "Supplies", "Tables"])
    segment = st.selectbox("Segment", ["Consumer", "Corporate", "Home Office"])
    ship_mode = st.selectbox("Ship Mode Encoded", ["Same Day", "First Class", "Second Class", "Standard Class"])
    delivery_time = st.sidebar.number_input("Delivery Time (in days)", min_value=0, max_value=8, value=8)

    data = {
        'Category': category,
        'Region': region,
        'Ship Mode Encoded': ship_mode,
        'Segment': segment,
        'Sub-Category': sub_category,
        'Delivery Time': delivery_time
    }

     # Convert the input data into a DataFrame
    features = pd.DataFrame(data, index=[0])

    # Convert categorical columns using one-hot encoding
    features = pd.get_dummies(features, columns=['Ship Mode Encoded', 'Category', 'Region', 'Sub-Category', 'Segment'])

    
    # Align the features with the training data (adding missing columns if necessary)
    features = features.reindex(columns=X_train.columns, fill_value=0)
    
    return features

input_df = user_input_features()

if st.button('Submit'):
    # Scale the user input
    input_scaled = scaler.transform(input_df)
    st.write(input_df)

    # Display the prediction results
    st.subheader('Input Parameters')

# Make prediction
if st.button("Predict Sales"):
    # Scale the input and make a prediction
    input_scaled = scaler.transform(input_df)
    prediction = predict_sales(input_scaled)
    st.subheader("Predicted Sales")
    st.write(f"${prediction[0][0]:.2f}")

