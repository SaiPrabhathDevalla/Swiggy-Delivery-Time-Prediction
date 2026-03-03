# Swiggy Delivery Time Prediction (ETA Prediction System)
A Machine Learning based regression system that predicts Estimated Time of Arrival (ETA) for food deliveries using rider, order, environmental, geospatial, and temporal features. This project simulates a real-world large-scale food delivery platform scenario similar to Swiggy, where accurate ETA prediction directly impacts customer satisfaction and operational efficiency.

# Problem Statement
* Predict delivery time (in minutes) at the time of order placement using:
* Rider information
* Restaurant & delivery location data
* Traffic and weather conditions
* Order timing features
* Operational constraints
* The goal is to build a production-ready ETA prediction model with strong generalization ability.

# Project Workflow
### Data Preprocessing
* Removed irrelevant columns (rider_id, order_date)
* Handled missing values
* Encoded categorical variables
* Applied feature scaling
* Performed train-test split (80% – 20%)

### Feature Engineering
#### Key features used in the model:
* Rider age and ratings
* Restaurant and delivery latitude/longitude
* Distance between locations
* Traffic and weather conditions
* Multiple deliveries
* Pickup preparation time
* Order day, month, and hour
* Weekend indicator
 City type and city name

### Model Building
#### Implemented multiple regression models:
* Linear Regression
* KNN Regressor
* Decision Tree Regressor
* Support Vector Regressor
* Random Forest Regressor
* XGBoost Regressor

### Model Selection
#### Models were compared using:
* R² Score
* Mean Squared Error (MSE)
* Cross-validation performance
* Best Model Selected: XGBoost Regressor
* High Testing R² Score
* Low MSE
* Strong generalization ability

### Model Performance
* Strong R² score on test dataset
* Low prediction error
* Balanced bias-variance tradeoff
* Stable cross-validation results

### Technologies Used
* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Pickle
