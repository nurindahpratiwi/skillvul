# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

# Load data from CSV files
customer_interactions = pd.read_csv("customer_interactions.csv")
purchase_history = pd.read_csv("purchase_history.csv", delimiter=";")
product_details = pd.read_csv("product_details.csv", delimiter=";")

# Exploration and Preprocessing handling
print(customer_interactions.info())
print(customer_interactions.describe())
print(customer_interactions.head())

print(purchase_history.info())
print(purchase_history.describe())
print(purchase_history.head())

print(product_details.info())
print(product_details.describe())
print(product_details.head())

# Drop unnamed columns
purchase_history = purchase_history.drop(columns=['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6'])
product_details = product_details.drop(columns=['Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6'])

# Check data
print(customer_interactions.head())
print(purchase_history.head())
print(product_details.head())

# Insert next 50 dummy data based on provided data
dummy_customer_interactions = pd.DataFrame({
    'customer_id': range(6, 56),
    'page_views': [25, 20, 30, 15, 22] * 10,
    'time_spent': [120, 90, 150, 80, 110] * 10
})

dummy_purchase_history = pd.DataFrame({
    'customer_id': [6, 6, 7, 7, 8, 8, 9, 9, 10, 10] * 5,
    'product_id': [101, 105, 102, 103, 104] * 10,
    'purchase_date': ['2023-01-01', '2023-01-05', '2023-01-02', '2023-01-03', '2023-01-04'] * 10
})

dummy_product_details = pd.DataFrame({
    'product_id': range(106, 156),
    'category': ['Electronics', 'Clothing', 'Home & Kitchen', 'Beauty', 'Electronics'] * 10,
    'price': [500, 50, 200, 30, 800] * 10,
    'ratings': [4.5, 3.8, 4.2, 4.0, 4.8] * 10
})

# Concatenate all dataframes
customer_interactions = pd.concat([customer_interactions, dummy_customer_interactions], ignore_index=True)
purchase_history = pd.concat([purchase_history, dummy_purchase_history], ignore_index=True)
product_details = pd.concat([product_details, dummy_product_details], ignore_index=True)

# Merge all dataframes on common columns
data = pd.merge(pd.merge(customer_interactions, purchase_history, on='customer_id'), product_details, on='product_id')

print(data.head())

# Build Predictive Model
# Feature selection
X = data[['page_views', 'time_spent', 'price', 'ratings']]
y = data['product_id']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Predictive Models
# RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_f1 = f1_score(y_test, rf_y_pred, average='weighted')
print(f"RandomForestClassifier F1 Score:{rf_f1*100}%")

# Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)
svm_f1 = f1_score(y_test, svm_y_pred, average='weighted')
print(f"SVM F1 Score:{svm_f1*100}%" )

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_y_pred = gb_model.predict(X_test)
gb_f1 = f1_score(y_test, gb_y_pred, average='weighted')
print(f"Gradient Boosting F1 Score:{gb_f1*100}%")

# RandomForestClassifier
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Support Vector Machine (SVM)
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Gradient Boosting
with open('gb_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)

# Load the selected model
def load_model(model_name):
    if model_name == "RandomForestClassifier":
        with open('rf_model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif model_name == "SVM":
        with open('svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        with open('gb_model.pkl', 'rb') as f:
            model = pickle.load(f)
    return model

# Function to preprocess input data for prediction
def preprocess_input(customer_id):
    # Merge customer interactions with purchase history or another DataFrame containing product_id
    merged_data = pd.merge(customer_interactions[customer_interactions['customer_id'] == customer_id], 
                           purchase_history, 
                           how='left', 
                           on='customer_id')
    
    # Merge with product details based on product_id
    merged_data = pd.merge(merged_data, product_details, how='left', on='product_id')
    
    # Select relevant features for prediction
    input_data = merged_data[['page_views', 'time_spent', 'price', 'ratings']].fillna(0)  # Fill missing values with 0, adjust as necessary
    return input_data

# Function to recommend top N products for a given customer ID using the selected model
def recommend_products(customer_id, top_n, model):
    # Preprocess input data
    input_data = preprocess_input(customer_id)
    
    # Perform predictions using the selected model
    predictions = model.predict(input_data)

    # Get top N product indices based on predictions
    top_indices = np.argsort(predictions)[::-1][:top_n]

    # Get top N product IDs
    top_product_ids = product_details.iloc[top_indices]['product_id'].tolist()
    top_product_names = product_details.iloc[top_indices]['category'].tolist()
    return top_product_ids, top_product_names

# Streamlit app
st.title('Product Recommendation System')

# User inputs for recommending products
customer_id = st.number_input('Enter Customer ID:', min_value=1, max_value=55, step=1)
top_n = st.number_input('Enter Top N Products:', min_value=1, max_value=10, step=1, value=5)

# Dropdown menu to select model
selected_model = st.selectbox("Select Model", ["RandomForestClassifier", "SVM", "Gradient Boosting"])

if st.button('Recommend'):
    # Load the selected model
    model = load_model(selected_model)
    
    # Perform product recommendation using the selected model
    recommendations_ids, recommendations_names = recommend_products(customer_id, top_n, model)
    st.write(f'Top {top_n} Recommended Products for Customer {customer_id} (Model: {selected_model}):')
    for i in range(len(recommendations_ids)):
        st.write(f"{recommendations_names[i]} (ID: {recommendations_ids[i]})")
