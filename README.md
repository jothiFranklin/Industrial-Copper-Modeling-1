# Industrial-Copper-Modeling
Python scripting, Data Preprocessing, EDA, Streamlit

The copper industry deals with less complex data related to sales and pricing.
However, this data may suffer from issues such as skewness and noisy data, which
can affect the accuracy of manual predictions. Dealing with these challenges manually
can be time-consuming and may not result in optimal pricing decisions. A machine
learning regression model can address these issues by utilizing advanced techniques
such as data normalization, feature scaling, and outlier detection, and leveraging
algorithms that are robust to skewed and noisy data.

Another area where the copper industry faces challenges is in capturing the leads. A
lead classification model is a system for evaluating and classifying leads based on
how likely they are to become a customer.

Data preprocessing and modeling pipeline steps:

1. Reading the data from a CSV file into a pandas DataFrame.
2. Dealing with incorrect data formats, converting some columns to appropriate data types (e.g., datetime, numeric).
3. Handling missing values by filling them for one column ('material_ref') and dropping rows with missing values for other columns.
4. Visualizing and transforming some numerical features using log transformations to address skewness.
5. Creating a Decision Tree Regressor to predict the 'selling_price_log' (log-transformed selling price) based on various features.
6. Saving the trained Decision Tree Regressor model, along with other preprocessing objects (scaler and one-hot encoders) using pickle.
7. Separating the dataset for a Decision Tree Classifier, which predicts the 'status' of sales as 'Won' or 'Lost'.
8. Encoding categorical variables and scaling numerical features.
9. Creating a Decision Tree Classifier for classifying sales status and evaluating its performance.
10. Saving the trained Decision Tree Classifier model, scaler, and one-hot encoders using pickle.

This appears to be a Streamlit web application for predicting the selling price and status of industrial copper based on user inputs. It utilizes machine learning models (specifically, DecisionTreeRegressor and another model for predicting the status) that have been previously trained and saved as pickle files.

Here's a brief overview of how the application works:

1. The application provides two tabs: "PREDICT SELLING PRICE" and "PREDICT STATUS."

2. In the "PREDICT SELLING PRICE" tab, the user can select various input parameters such as "Status," "Item Type," "Country," "Application," and "Product Reference." The user also needs to enter values for "Quantity Tons," "Thickness," "Width," and "Customer ID." Once the user clicks the "PREDICT SELLING PRICE" button, the application will use the input data and the previously trained model to predict the selling price, which will be displayed on the screen.

3. In the "PREDICT STATUS" tab, the user can enter the same input parameters as in the previous tab but also needs to provide a "Selling Price." Clicking the "PREDICT STATUS" button will use the input data and another previously trained model to predict the status of the industrial copper (whether it is "Won" or "Lost").

It's important to note that the application uses specific pickle files to load the trained models, scalers, and encoders. These pickle files must exist and contain the necessary components for the application to function correctly.

Overall, this Streamlit app provides a user-friendly interface for predicting the selling price and status of industrial copper based on the input provided by the user.
