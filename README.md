# Cryptocurrency Price Prediction - Machine Learning Models

## Table of Contents

1. [Overview](#overview)
2. [Data Source](#data-source)
3. [Algorithms Used](#algorithms-used)
   - [Linear Regression](#linear-regression)
   - [Decision Trees](#decision-trees)
   - [Random Forest](#random-forest)
   - [Support Vector Machines (SVM)](#support-vector-machines-svm)
   - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
   - [Neural Networks](#neural-networks)
4. [Final Model](#final-model)
5. [Real-Time Data and Predictions](#real-time-data-and-predictions)
6. [Website](#website)
7. [Performance Comparison](#performance-comparison)
8. [Installation and Setup](#installation-and-setup)
9. [Contributing](#contributing)
10. [License](#license)

## Overview

This project aims to predict cryptocurrency prices (bitcoin,binance,litecoin) using various machine learning algorithms. The goal was to evaluate and compare different models to identify the most effective approach for accurate predictions.

## Data Source

The dataset used for this project is sourced from Yahoo Finance and consists of real-time data. This includes up-to-date historical prices and features relevant for predicting cryptocurrency prices, allowing the model to generate predictions for today as well.

## Algorithms Used

### Linear Regression
- **Description**: A simple algorithm that assumes a linear relationship between input features and the target variable.
- **Implementation**: Standard Linear Regression model trained on the dataset to predict cryptocurrency prices.

### Decision Trees
- **Description**: Builds a model in the form of a tree structure, with each node representing a decision based on feature values.
- **Implementation**: Decision Tree model trained to capture complex interactions between features.

### Random Forest
- **Description**: An ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce overfitting.
- **Implementation**: Random Forest model used to aggregate predictions from multiple decision trees.

### Support Vector Machines (SVM)
- **Description**: Constructs hyperplanes in a high-dimensional space to classify or regress data.
- **Implementation**: SVM applied to predict cryptocurrency prices by finding the optimal separating hyperplane.

### K-Nearest Neighbors (KNN)
- **Description**: Classifies or predicts the value of a point based on the values of its k-nearest neighbors.
- **Implementation**: KNN algorithm used to predict prices by analyzing the proximity of data points.

### Neural Networks
- **Description**: Complex models inspired by the human brain, capable of learning non-linear relationships in data.
- **Implementation**: Neural Networks used to capture intricate patterns in the data for price prediction.

## Final Model

After evaluating the performance of various algorithms, the **Feedforward Neural Network** was chosen as the final model. It demonstrated superior efficiency and accuracy compared to other methods. The Feedforward Neural Network effectively captures complex patterns in the data, making it well-suited for real-time price prediction.

## Real-Time Data and Predictions

The application uses real-time data from Yahoo Finance, allowing it to provide predictions based on today's data. This ensures that users receive the most current and relevant predictions.

## Website

The prediction results and model interface are presented through a web application built using **Streamlit**. Streamlit provides an interactive and user-friendly platform for visualizing predictions and exploring the data.

## Performance Comparison

The performance of the algorithms was evaluated using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared values. The comparative analysis highlights the effectiveness of each algorithm, with Feedforward Neural Network emerging as the most efficient.

## Installation and Setup

To get started with this project, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ShubhrangiD/crypto-price-prediction.git
   cd crypto-price-prediction
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**
   - Ensure your dataset is in the appropriate directory.
   - Start the Streamlit app:
     ```bash
     streamlit run app.py
     ```

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request with your changes. Include relevant tests and documentation as needed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
