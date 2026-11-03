# Real Estate Investment Advisor

A Streamlit-based machine learning app for evaluating residential property investment potential in India. The app predicts whether a property is likely to be a good investment and estimates its likely value after five years using a trained classifier and regressor.

## Overview

This project combines:

- Property input form for price, area, bedrooms, location score, and more
- Machine learning-based investment classification
- 5-year future price estimation
- Interactive market insights and filtered property analysis
- Model performance dashboard with feature importance charts

The app is designed to help buyers and investors quickly evaluate property opportunities using historical market patterns and model-based predictions.

## Features

- Predict whether a property is a good investment
- Estimate future property value after 5 years
- Show expected annualized growth and financial impact
- Visualize projected price trend curves
- Explore filtered market data for comparable properties
- View model metrics and important prediction features

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Plotly
- Joblib
- MLflow

## Project Structure

```text
.
├── app.py                          # Main Streamlit application
├── best_classifier.pkl            # Trained classification model
├── regressor.pkl                  # Trained regression model
├── merged_houseprice_regression_classification-Regression+Class.csv
├── india_housing_prices.csv       # Additional housing dataset
├── house.ipynb                    # Notebook for analysis/model exploration
├── requirments.txt                # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

## Prerequisites

- Python 3.10 or later
- pip
- A local environment for installing project dependencies

## Installation

1. Clone the repository:

```bash
git clone https://github.com/prathamesh9730/Real-Estate-Investment-Advisor.git
cd Real_estate
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

3. Install dependencies:

```bash
pip install -r requirments.txt
```

## Run the App

Start the Streamlit application with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (typically http://localhost:8501).

## How It Works

The application loads the dataset and trains or loads a classifier and regressor when needed. Based on property features such as:

- area in sqft
- bedrooms
- bathrooms
- floors
- year built
- location score
- school score
- current price

it predicts:

- whether the investment is likely to be good
- the expected future price after 5 years

## Notes

- The app uses the dataset file located near `app.py` and expects the model files to be present in the same directory.
- Prediction outputs are for learning and exploratory purposes and should not be treated as formal financial advice.
- The project includes both regression and classification workflows for property valuation and investment suitability.

## License

This project does not currently declare a license. If you plan to share or distribute it, add an appropriate open-source license.

## Author

prathamesh9730
