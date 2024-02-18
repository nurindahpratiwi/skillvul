# Product Recommendation System

Welcome to the Product Recommendation System! This system provides personalized product recommendations based on customer interactions and purchase history.

## Installation

To install the required dependencies, simply run:

```bash
pip install -r requirements.txt
```

## Running Locally

You can run the system locally using Streamlit. Use the following command:

```bash
streamlit run predict.py
```

## Online Demo

Alternatively, you can access the system through the following URL:

[Product Recommendation System](https://skillvul-technicaltest.streamlit.app/)

## Usage

1. **Input Customer ID:** Enter the customer ID for which you want to receive product recommendations. The system will retrieve relevant information based on this ID.

2. **Select Model:** Choose the machine learning model you want to use for generating recommendations. You can select from RandomForestClassifier, SVM, or Gradient Boosting.

3. **Recommend:** Click the "Recommend" button to generate top N recommended products for the selected customer ID.

### How It Works

The system utilizes machine learning algorithms trained on customer interactions and purchase history data to predict the most suitable products for a given customer. Users can input a customer ID, and the system will leverage the selected model to generate personalized recommendations. These recommendations are based on the customer's past interactions and purchase behavior, ensuring relevance and accuracy.
