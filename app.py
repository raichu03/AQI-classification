import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import streamlit as st

# Load the dataset
data = pd.read_csv('/home/pranjal/Downloads/aq_kathmandu_us-diplomatic-post_embassy_kathmandu.csv')

# Convert 'utc' to datetime
data['utc'] = pd.to_datetime(data['utc'])

# Label the data based on PM2.5 values
def label_pm25(value):
    if value <= 12:
        return 'Good'
    elif value <= 35:
        return 'Moderate'
    elif value <= 55:
        return 'Unhealthy for Sensitive Groups'
    elif value <= 150:
        return 'Unhealthy'
    elif value <= 250:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

# Filter only PM2.5 data and apply labeling
pm25_data = data[data['parameter'] == 'pm25']
pm25_data['label'] = pm25_data['value'].apply(label_pm25)

# Feature extraction
pm25_data['hour'] = pm25_data['utc'].dt.hour
pm25_data['dayofweek'] = pm25_data['utc'].dt.dayofweek
pm25_data['month'] = pm25_data['utc'].dt.month

# Prepare features and labels
X = pm25_data[['hour', 'dayofweek', 'month']]
y = pm25_data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Streamlit deployment
def main():
    st.title('Air Quality Classification')

    # Input features
    hour = st.number_input('Hour', min_value=0, max_value=23, value=12)
    dayofweek = st.number_input('Day of the Week', min_value=0, max_value=6, value=0)
    month = st.number_input('Month', min_value=1, max_value=12, value=1)

    # Predict button
    if st.button('Predict Air Quality Level'):
        input_data = pd.DataFrame({'hour': [hour], 'dayofweek': [dayofweek], 'month': [month]})
        prediction = model.predict(input_data)
        st.write(f'The predicted air quality level is: {prediction[0]}')

if __name__ == '__main__':
    main()
