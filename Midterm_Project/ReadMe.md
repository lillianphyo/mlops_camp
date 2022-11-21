# Loan Defaults Predictions

### Problem Statement

Objective is to build a machine learning model to figure out customer retention rate is good or not in specific amount of time.

### Dataset

The train and test set contains the different attributes related to demographic and loan information of the applicants as age, profession, no. of active loans, and loan default in previous loans. The training set contains the target variable loan_default.

| Variable | Description |
|-|-|
|ID| Unique identifier of customers |
|BillingType| service subscribtion charges for periodically|
|Status| Relationship between customers and organization (partner or !partner for example)|
|Category | Customer Types |
|DateAdd| Customer acquired date|
|LastOnline| Customers' last connect time|
|MrrTotal| Total monthly recurring revenue based on ARPU average revenue per users |
|Quantity|No of active service|
|ServicesStart Date|First day of service active|
|ServicesEnd Date|Last day of service active|
|ServicesStatus (Targeted Variable) | Status of services |
|ServiceName| Service category |
|Price| Service subscription price|
|Locations| Location of customers|

### 1. Exploratory Data Analysis
Statistical data analysis and Data Visualization performed on data to find if any outlier or missing data are present.

### 2. Data Preprocessing

***age*** - Binning or Discretization is performed on the age feature for better analysis.

***education*** - Missing values in the education column are filled with its `mode` value.

The ***loan_amount***, ***asset_cost***, ***no_of_loans***, ***no_of_curr_loans***, ***last_delinq_none*** features has a skewed distribution with outliers. After removing the outliers, Normalization is performed on data using `StandardScaler`.

### 3. Imbalanced Data.
The data is highly imbalanced; the number of customers who default is significantly smaller than the number of customers who did not. I tried different approaches to handle imbalanced data using the `imblearn` library.

* **SMOTE oversampling**: Using SMOTE oversampling method gives the best results on training data but overfits the model.

* **RandomUnderSampler**: *RandomUnderSampler* does not perform well on both training and testing data.

* **Using Both (SMOTE and RandomUnderSampler)**: Resampled data first with SMOTE oversampling and then with RandomUnderSampler, Using both methods on data gives the best score and hence selected for model training.

### 4. Model Training
I tried several base models such as `LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier` on balanced data. Among these models, `LogisticRegression` gives the best macro f1 score. Therefore this model is used for final submission.


