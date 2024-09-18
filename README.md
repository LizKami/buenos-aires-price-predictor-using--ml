# buenos-aires-price-predictor-using-ml
The goal of this project is to develop a machine learning model capable of predicting apartment prices in Buenos Aires using linear regression with a focus on apartments that cost less than $400000.

**Problem Statement**

In the real estate market, accurate price predictions are crucial for buyers, sellers, and realtors. Buenos Aires, Argentina, has a dynamic and competitive housing market, but predicting apartment prices remains a challenge due to various factors such as location, size, and market trends. Traditional methods of estimating prices may not capture the complexity of the data, resulting in inaccurate forecasts.

The goal of this project is to develop a machine learning model capable of predicting apartment prices in Buenos Aires using linear regression. This model will leverage data wrangling and visualization techniques to preprocess real estate data, handle missing values, and encode categorical features. The project will focus on creating a robust data pipeline to minimize overfitting and enhance model performance. Additionally, a dynamic dashboard will be developed to allow users to interact with the model and gain insights into price predictions.

This predictive tool aims to provide valuable insights to various stakeholders in the Buenos Aires real estate market, offering more accurate price estimates and aiding decision-making.


**Mehodology**

Based on the python script provided, I took the following steps to achive the set out  goal:

1. Data Cleaning and Preprocessing:

The script displays the wrangle function used to clean and preprocess real estate data.

Key preprocessing steps include:

-Filtering apartments in "Capital Federal" with a price below $400,000.

-Removing outliers based on the surface_covered_in_m2 variable.

-Splitting the lat-lon column into separate latitude and longitude columns.

-Dropping features with high null counts, low and high cardinality categorical variables, leaky columns, and columns with multicollinearity.

2. Data Concatenation and Exploration:

-Data from multiple CSV files is concatenated into a single DataFrame.

-A correlation heatmap of numerical features (excluding the target variable) is plotted to explore relationships between features.

3. Model Building and Evaluation:

- A linear regression model (Ridge regression) is built using a pipeline that includes one-hot encoding, imputation of missing values, and ridge regression.
- Baseline mean absolute error (MAE) is calculated to compare the model’s performance. The baseline MAE is based on the mean of the target variable.
- The model’s performance on the training data is evaluated using MAE.
- Predictions are made on test data, and the results are displayed.

4. Model Deployment:

- A function make_prediction is created to allow predictions based on user-provided input.

- An interactive dashboard using ipywidgets allows users to input parameters and see the predicted apartment price dynamically.
