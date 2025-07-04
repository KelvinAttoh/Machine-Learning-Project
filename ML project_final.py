#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import root_mean_squared_error, accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve


# ## Query the data

# In[107]:


glaucoma = pd.read_csv('glaucoma_dataset.csv')
glaucoma.shape


# In[108]:


# Load the data

glaucoma.head()


# ## Data preprocessing

# #### Split dataset into numerical and categorical

# In[109]:


num_data = glaucoma.select_dtypes('number')
num_data


# In[110]:


cat_data = glaucoma.select_dtypes('object')
cat_data


# #### Dealing with inconsistencies for categorical data

# In[111]:


cat_data.columns


# In[112]:


print(cat_data['Visual Acuity Measurements'].unique())

# Fix inconsistencies for Visual Acuity Measurements column

cat_data['Visual Acuity Measurements'].replace('20/20', 'LogMAR 0.0', inplace = True) # Change Snellen measurements to LogMAR
cat_data['Visual Acuity Measurements'].replace('20/40', 'LogMAR 0.3', inplace = True)


# In[113]:


print(cat_data['Medical History'].unique())

# Fix inconsistencies for Visual Acuity Measurements column

cat_data['Medical History'].replace(np.nan, 'No history', inplace = True) # Change missing value to No record


# #### Deal with unnecessary columns for categorical data

# In[114]:


cat_data.drop(columns = ['Medication Usage', 'Visual Symptoms'], axis = 1, inplace = True)
cat_data


# #### Deal with missing values for categorical data

# In[115]:


cat_data.isna().any()


# In[116]:


# Confirm the number of missing values in each categorical data

cat_data.isna().sum()

print("No missing values for categorical data")


# #### Deal with duplicates for categorical data

# In[117]:


# Confirm whther there are duplicates for categorical data
print(cat_data.duplicated().any())

# The number of duplicates
print("Number of duplicated records in categorical data: {}".format(cat_data.duplicated().sum()))


# #### Deal with unnecessary data for numerical data

# In[118]:


num_data.drop(columns = 'Patient ID', axis = 1, inplace = True)


# #### Deal with duplicates for numerical data

# In[119]:


# Confirm whther there are duplicates for categorical data
print(num_data.duplicated().any())

# The number of duplicates
print("Number of duplicated records in categorical data: {}".format(num_data.duplicated().sum()))

# No duplicates


# #### Deal missing values for numerical data

# In[120]:


num_data.columns


# In[121]:


# Confirm the number of missing values in each categorical data
print(num_data.isna().any())

# drop 'Unnamed: 20' column
num_data.drop(columns = 'Unnamed: 20', axis = 1, inplace = True)


# #### Deal with outliers for numerical data

# In[122]:


# Remove outliers beyond 95% interval in normal distribution

num_outliers = {}

for col in num_data.columns:
    Q1 = num_data[col].quantile(0.25)
    Q3 = num_data[col].quantile(0.75)
    IQR = Q3 - Q1
    # define bounds within 95% interval
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR

    # identify outliers
    outliers = num_data[(num_data[col] < lower_limit) | (num_data[col] > upper_limit)]
    num_outliers[col] = outliers

    # print outlier info
    print("\n\nThe outliers indices for {} are:\n{}".format(col, list(set(outliers.index))))
    print("The number of outliers removed from {} column are: {}".format(col, len(outliers)))

    # remove outliers from num_data
    num_data = num_data[(num_data[col] >= lower_limit) & (num_data[col] <= upper_limit)]

# num_data is now cleaned

print("\n\nNo outliers detected in each column for numerical data")


# #### Concatenate numerical and categorical data

# In[123]:


cleaned_data = pd.concat([num_data, cat_data], axis = 1)
cleaned_data


# #### Encode categorical values to boolean values

# In[124]:


cleaned_data['Diagnosis'].value_counts()


# In[125]:


# Label Encoding on selected columns

le_columns = ['Gender', 'Family History', 'Cataract Status', 'Angle Closure Status', 'Diagnosis']

le = LabelEncoder()

for col in le_columns:
    cleaned_data[col] = le.fit_transform(cleaned_data[col])


# In[126]:


# One Hot Encoding on selected columns

OHE = ['Visual Acuity Measurements', 'Medical History', 'Glaucoma Type']

# Generate dummy variables for selected columns and drop the first category
cleaned_data = pd.get_dummies(cleaned_data, columns = OHE, drop_first=True)


# #### View cleaned dataset

# In[127]:


pd.set_option('display.max_columns', None)
cleaned_data


# ### Visualize dataset

# #### Create boxplot 

# In[23]:


# Specific numeric columns of interest
numeric_cols = [
    'Age', 'Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)',
    'Visual Field Test Results (Sensitivity)', 'Visual Field Test Results (Specificity)',
    'OCT Results (RNFL Thickness)', 'OCT Results (GCC Thickness)',
    'Retinal Volume', 'Macular Thickness', 'Pachymetry'
]

num_cols = 2
num_rows = 5

plt.figure(figsize=(9, 18))

for idx, col in enumerate(numeric_cols):
    plt.subplot(num_rows, num_cols, idx + 1)
    sns.boxplot(data=cleaned_data, x='Diagnosis', y=col, palette='Set2')
    plt.title(f'{col} by Diagnosis', fontsize=10)
    plt.xlabel('Diagnosis', fontsize = 12)
    plt.ylabel(col)
    plt.xticks([0, 1], ['Glaucoma', 'Non-Glaucoma'])

plt.suptitle('Numeric Feature Distributions by Diagnosis', fontsize=12, y=1)
plt.tight_layout()
plt.show()


# #### Create histogram

# In[24]:


plt.figure(figsize=(20, 15))

num_cols = 3
num_rows = 3 

categorical_columns = [
    'Gender', 'Family History', 'Cataract Status', 'Angle Closure Status',
    'Visual Acuity Measurements_LogMAR 0.1', 'Visual Acuity Measurements_LogMAR 0.3',
    'Medical History_Glaucoma in family']

for index, col in enumerate(categorical_columns):
    plt.subplot(num_rows, num_cols, index + 1)
    
    sns.countplot(data=cleaned_data, x=col, hue='Diagnosis', palette='colorblind')
    
    plt.xlabel(col, fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.title(f'{col} by Diagnosis', fontsize=15)
    plt.tick_params(labelsize = 12)
    plt.xticks(ha='right')

plt.tight_layout()
plt.show()


# In[25]:


plt.figure(figsize=(20, 15))

num_cols = 3
num_rows = 3 

categorical_columns = [
    'Medical History_Hypertension', 'Medical History_No history', 'Glaucoma Type_Congenital Glaucoma',
    'Glaucoma Type_Juvenile Glaucoma', 'Glaucoma Type_Normal-Tension Glaucoma',
    'Glaucoma Type_Primary Open-Angle Glaucoma', 'Glaucoma Type_Secondary Glaucoma'
]

for index, col in enumerate(categorical_columns):
    plt.subplot(num_rows, num_cols, index + 1)
    
    sns.countplot(data=cleaned_data, x=col, hue='Diagnosis', palette='husl')
    
    plt.xlabel(col, fontsize=15)
    plt.ylabel('Count', fontsize=15)
    plt.title(f'Distribution of {col} by Diagnosis', fontsize=12)
    plt.tick_params(labelsize = 12)
    plt.xticks(ha='right')

plt.tight_layout()
plt.show()


# In[26]:


# Select numeric columns excluding the target
numeric_cols = cleaned_data.select_dtypes('number').columns.drop('Diagnosis')

# Define grid size based on number of numeric columns
num_rows = 5
num_cols = 3

plt.figure(figsize=(18, 25))

for index, col in enumerate(numeric_cols):
    plt.subplot(num_rows, num_cols, index + 1)

    sns.histplot(data=cleaned_data, x=col, hue='Diagnosis', kde=True,
                 palette={0: 'teal', 1: 'salmon'}, bins=30, multiple='stack')

    plt.title(f'{col}', fontsize=10)
    plt.xlabel('')
    plt.ylabel('')

    # Optional: remove legends to declutter
    if index != 0:
        plt.legend([], [], frameon=False)

plt.suptitle('Histograms of Numeric Features by Diagnosis (Glaucoma vs Non-Glaucoma)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# #### Create correlation heatmap

# In[27]:


plt.figure(figsize=(25, 18))

# Compute correlation matrix
correlation_matrix = cleaned_data.corr()

# Create heatmap
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Map for Features', fontsize = 22)
plt.tick_params(labelsize=15)

# Show plot
plt.show()


# In[28]:


number = cleaned_data.select_dtypes('number')

number.columns


# ## Data Modeling

# ### Data Subsampling and Train-Test Split

# We subsample 80% of the full dataset (while preserving class imbalance) and then split it into training and testing sets.

# In[29]:


X_full = cleaned_data.drop(columns = 'Diagnosis', axis = 1)  # Extract features
y_full = cleaned_data['Diagnosis'].values  # Extract target (0 = Glaucoma, 1 = No Glaucoma)

print("Full dataset class distribution:", np.bincount(y_full))


# In[30]:


# Subsample 20% of the dataset while preserving class imbalance
X, _, y, _ = train_test_split(X_full, y_full, train_size=0.8, random_state=42, stratify=y_full)
print("Subsampled 80% dataset class distribution:", np.bincount(y))

# Split subsampled data: 80% train, 20% test, stratified to maintain imbalance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nTraining class distribution (80% subsample):", np.bincount(y_train))


# In[31]:


# scale dataset

scaler = StandardScaler()

# Scale features (X_train)
X_train_scaled = scaler.fit_transform(X_train)  # Scales features to range (0,1)



# ### Introduce RandomForest Model

# In[32]:


RF_clf = RandomForestClassifier(n_estimators=1000, random_state=42)
RF_clf.fit(X_train_scaled, y_train)


# In[33]:


# Checking training and testing accuracy for imbalance dataset

train1_accuracy = accuracy_score(y_train, RF_clf.predict(X_train_scaled))

test1_accuracy = accuracy_score(y_test, RF_clf.predict(X_test))

print(f"train accuracy for imbalanced dataset : {train1_accuracy :.2f}")
print(f"test accuracy for imbalanced dataset : {test1_accuracy :.2f}")

mse = accuracy_score(y_test, RF_clf.predict(X_test))
print("Model Mean Squared Error:", mse)


# In[34]:


# Plot accuracy score

plt.figure(figsize=(6, 4))
plt.bar(["Training Accuracy", "Test Accuracy"], [train1_accuracy, test1_accuracy], color = ['teal', 'sienna'])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Overfitted RandomForest Accuracy")
plt.show()


# ### Balance the dataset

# In[35]:


# Initialize RandomOverSampler
ROS = RandomOverSampler(random_state=42)

# Apply oversampling
X_train_sampled, y_train_sampled = ROS.fit_resample(X, y)


# ### Hyperparameter Tuning

# from sklearn.model_selection import GridSearchCV
# 
#  Define parameter grid
# param_grid = {
#     'n_estimators' : [1000, 2000, 3000],
#     'max_depth': [10, 12, 14, 16],
#     'min_impurity_decrease': [0.0001, 0.001, 0.1, 1],
#     'ccp_alpha' : [0.0001, 0.001, 0.1]
# }
# 
#  Perform Grid Search
# grid_search = GridSearchCV(RandomForestClassifier(class_weight = 'balanced', random_state=42), param_grid, cv=5)
# grid_search.fit(X_train_sampled, y_train_sampled)
# 
#  Best parameters
# print("Best Parameters:", grid_search.best_params_)
# 
#  Evaluate model with best parameters
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)
# mse = accuracy_score(y_test, y_pred)
# print("Best Model Mean Squared Error:", mse)

#  Introduce the machine learning model for balanced dataset with hyperparameter tuning
# 
# RF_clf = RandomForestClassifier(ccp_alpha = 0.001, max_depth = 12, min_impurity_decrease = 0.0001, n_estimators = 1000, class_weight = 'balanced', random_state=42)
# RF_clf.fit(X_train_sampled, y_train_sampled)

# In[36]:


# Introduce the machine learning model for balanced dataset with hyperparameter tuning

RF_clf = RandomForestClassifier(max_depth = 10, n_estimators=500, class_weight = 'balanced', random_state=42)
RF_clf.fit(X_train_sampled, y_train_sampled)


#  Introduce the machine learning model for balanced dataset with hyperparameter tuning
# 
# RF_clf = RandomForestClassifier(max_depth = 14, n_estimators=500, class_weight = 'balanced', ccp_alpha = 0.001, criterion = 'entropy', min_impurity_decrease = 0.0001, random_state=42)
# RF_clf.fit(X_train_sampled, y_train_sampled)

# In[37]:


# Set up 10-Fold Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Perform Cross-Validation
cv_scores = cross_val_score(RF_clf, X_train_sampled, y_train_sampled, cv=skf, scoring='accuracy')

# Print results
print(f'Cross-validation Accuracy Scores: {cv_scores}')
print(f'Mean Accuracy: {cv_scores.mean():.4f}')
print(f'Standard Deviation: {cv_scores.std():.4f}')


# In[38]:


# Checking training and testing accuracy for balance dataset

train2_accuracy = accuracy_score(y_train_sampled, RF_clf.predict(X_train_sampled))

test2_accuracy = accuracy_score(y_test, RF_clf.predict(X_test))

print(f"train accuracy for balanced dataset : {train2_accuracy :.2f}")
print(f"test accuracy for balanced dataset : {test2_accuracy :.2f}")


# ### Feature extraction

# In[39]:


# Explicitly convert NumPy array to DataFrame with original column names

X_df = pd.DataFrame(X, columns=X_train.columns)

# Extract feature importance using the correct column names
feature_importance = pd.Series(RF_clf.feature_importances_, index=X_df.columns)

# Display sorted feature importance
print(feature_importance.sort_values(ascending=False))


# In[40]:


# Visualize feature importance

plt.figure(figsize = (8, 6))
feature_importance.sort_values(ascending = False).plot(kind = 'bar', color = 'sienna')
plt.title('Feature importance')
plt.xlabel('Features')
plt.ylabel('Importance score')
plt.show()


# ### Feature selection

# In[41]:


from sklearn.feature_selection import RFE

# Ensure X_train_sampled is a DataFrame and matches y_train_sampled
# Rebuild X_df with correct number of rows and column names
X_df = pd.DataFrame(X_train_sampled, columns=X_train.columns)  # Use original feature names from X_train

# Set up RandomForest and RFE
FS = RFE(estimator=RF_clf, n_features_to_select=10)
FS.fit(X_df, y_train_sampled)

# Get selected feature names
selected_features = X_df.columns[FS.support_].tolist()
print("\nSelected Features:")
print(selected_features)

# Create training and test sets with selected features
X_train_selected = X_df[selected_features]
X_test_selected = pd.DataFrame(X_test, columns=X_train.columns)[selected_features]

# Re-train model on selected features
RF_clf.fit(X_train_selected, y_train_sampled)

# Evaluate
train_acc = accuracy_score(y_train_sampled, RF_clf.predict(X_train_selected))
test_acc = accuracy_score(y_test, RF_clf.predict(X_test_selected))

print(f"\nTraining Accuracy with selected features: {train_acc:.2f}")
print(f"Testing Accuracy with selected features: {test_acc:.2f}")


# In[42]:


# Evaluate the RandomForest model

y_pred = RF_clf.predict(X_test_selected)


# In[43]:


# Accuracy of the RandomForest model

A_score1 = accuracy_score(y_test, y_pred)
print(f'Accuracy of the RandomForest model: {A_score1 * 100 : .0f}%')


# In[44]:


# Performance of the RandomForest model

train_performance = root_mean_squared_error(y_train_sampled, RF_clf.predict(X_train_selected))

test_performance = root_mean_squared_error(y_test, y_pred)

print(f'Root mean squared error of the RandomForest model train performance: {train_performance : .2f}')
print(f'Root mean squared error of the RandomForest model test performance: {test_performance: .2f}')


# In this case, the Random Forest model has an RMSE of 0.30 on the training set and 0.30 on the test set, suggesting that the model generalizes well to unseen data and is not overfitting.

# In[45]:


# Plot accuracy score

plt.figure(figsize=(6, 4))
plt.bar(["Training Accuracy", "Test Accuracy"], [train_acc, test_acc], color = ['pink', 'skyblue'])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("RandomForest Accuracy after balancing dataset")
plt.show()


# In this case, the Random Forest model has an RMSE of 0.30 on the training set and 0.30 on the test set, suggesting that the model generalizes well to unseen data and is not overfitting.

# In[46]:


# ROC curve score

roc_auc_score(y_test,y_pred)


# In[47]:


# Visualize ROC curve for RandomForest model

plt.figure(figsize=(8,6))

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.1])
plt.ylim([0.0, 1.1])
plt.title('ROC curve for Glaucoma disease classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# In[48]:


# Classification report for RandomForest model

c_report1 = classification_report(y_test, y_pred)
print(c_report1)


# In[49]:


# Build confusion matrix for RandomForest model

CM1 = confusion_matrix(y_pred, y_test)

plt.figure(figsize = (4, 3))
sns.heatmap(CM1, annot = True, fmt = 'g')
plt.title('Confusion Matrix for RandomForest model')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# ### Introduce Logistic Regression Model

# In[50]:


# import libraries

from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')
logReg.fit(X_train_selected, y_train_sampled)


# In[51]:


# Set up 10-Fold Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Perform Cross-Validation
cv_scores = cross_val_score(logReg, X_train_selected, y_train_sampled, cv=skf, scoring='accuracy')

# Print results
print(f'\nCross-validation Accuracy Scores: {cv_scores}')
print(f'\nMean Accuracy: {cv_scores.mean():.4f}')
print(f'\nStandard Deviation: {cv_scores.std():.4f}\n')


# In[52]:


# predict the X_test

y_predd = logReg.predict(X_test_selected)


# In[53]:


# Accuracy of the Logistic Regression model

A_score2 = accuracy_score(y_test, y_predd)
print(f'Accuracy of the Logistic Regression model: {A_score2 * 100 : .0f}%')


# #### LASSO Regularization - Logistic Regression model

# In[54]:


# Regularization strengths
C_values = np.logspace(-2, 2, 10)

# Track coefficients and validation scores
coefficients_l1 = []
mean_scores = []

# Fit L1 logistic regression for each C
for C in C_values:
    logReg_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=500, random_state=42, class_weight='balanced')
    logReg_l1.fit(X_train_selected, y_train_sampled)
    
    # Store the coefficients
    coefficients_l1.append(np.linalg.norm(logReg_l1.coef_[0]))
    
    # Use cross-validation to track the accuracy
    scores = cross_val_score(logReg_l1, X_train_selected, y_train_sampled, cv=10, scoring='accuracy')
    mean_scores.append(np.mean(scores))

# Identify the optimal C based on the highest accuracy
optimal_index_l1 = np.argmax(mean_scores)
optimal_C_l1 = C_values[optimal_index_l1]

# Get the corresponding coefficient magnitude at optimal C
optimal_coefficient_l1 = coefficients_l1[optimal_index_l1]

# Plotting the coefficients
plt.figure(figsize=(12, 6))

# L1 Regularization plot
plt.plot(C_values, coefficients_l1, label='L1 Norm', marker='x', color='b')
plt.xscale('log')
plt.xlabel('C (Inverse of Regularization Strength)')
plt.ylabel('Coefficient Magnitude')
plt.title('L1 Regularization Coefficient Magnitude')

# Add a vertical line at the C corresponding to the optimal coefficient
plt.axvline(x=optimal_C_l1, color='r', linestyle='--', label=f'Optimal C = {optimal_C_l1:.4f}')
plt.scatter(optimal_C_l1, optimal_coefficient_l1, color='r', zorder=5, label=f'Optimal Coefficient = {optimal_coefficient_l1:.4f}')

plt.legend()
plt.show()


# In[55]:


# Optimal Coefficient Value and Magnitude

print(f"Optimal Coefficient Value (L1 regularization): {optimal_C_l1:.2f}")
print(f"Optimal Coefficient Magnitude: {optimal_coefficient_l1:.2f}")


# In[56]:


# Set up 10-Fold Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Perform Cross-Validation
cv_scores = cross_val_score(logReg_l1, X_train_selected, y_train_sampled, cv=skf, scoring='accuracy')

# Print results
print(f'Cross-validation Accuracy Scores: {cv_scores}')
print(f'Mean Accuracy: {cv_scores.mean():.2f}')
print(f'Standard Deviation: {cv_scores.std():.4f}')


# In[57]:


# Tune hyperparameter on Logistic Regression for L1 regularization

logReg_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=optimal_C_l1, l1_ratio = optimal_coefficient_l1, max_iter=500, class_weight='balanced', random_state = 42)
logReg_l1.fit(X_train_selected, y_train_sampled)


# In[58]:


# predict the X_test

y_predd_l1 = logReg_l1.predict(X_test_selected)


# In[59]:


# Track accuracy and coefficients
accuracies_l1 = []

A_score_l1 = accuracy_score(y_test, y_predd_l1)
accuracies_l1.append(A_score_l1)
coefficients_l1.append(np.abs(logReg_l1.coef_[0]))

# Accuracy of the L2 Logistic Regression model

print(f'Accuracy of the L1 Logistic Regression model: {max(accuracies_l1) * 100:.0f}%')


# In[60]:


# ROC curve score L1 Logistic Regression model

roc_auc_score(y_test,y_predd_l1)


# In[61]:


# Visualize ROC curve for L1 Logistic Regression model

plt.figure(figsize=(8,6))

fpr, tpr, thresholds = roc_curve(y_test, y_predd_l1)
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.1])
plt.ylim([0.0, 1.1])
plt.title('ROC curve for Glaucoma disease classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# #### Ridge Regularization - Logistic Regression model

# In[62]:


# Regularization strengths
C_values = np.logspace(-4, 4, 10)

# Track coefficients and validation scores
coefficients_l2 = []
mean_scores = []

# Fit L2 logistic regression for each C
for C in C_values:
    logReg_l2 = LogisticRegression(penalty='l2', solver='liblinear', C=C, max_iter=500, random_state=42, class_weight='balanced')
    logReg_l2.fit(X_train_selected, y_train_sampled)
    
    # Store the coefficient magnitudes
    coefficients_l2.append(np.linalg.norm(logReg_l2.coef_[0]))
    
    # Use cross-validation to track the accuracy
    scores = cross_val_score(logReg_l2, X_train_selected, y_train_sampled, cv=10, scoring='accuracy')
    mean_scores.append(np.mean(scores))

# Identify the optimal C based on the highest accuracy
optimal_index_l2 = np.argmax(mean_scores)
optimal_C_l2 = C_values[optimal_index_l2]

# Get the corresponding coefficient magnitude at optimal C
optimal_coefficient_l2 = coefficients_l2[optimal_index_l2]

# Plotting the coefficients
plt.figure(figsize=(12, 6))

# L2 Regularization plot
plt.plot(C_values, coefficients_l2, label='L2 Norm', marker='x', color='b')
plt.xscale('log')
plt.xlabel('C (Inverse of Regularization Strength)')
plt.ylabel('Coefficient Magnitude')
plt.title('L2 Regularization Coefficient Magnitude')

# Add a vertical line at the C corresponding to the optimal coefficient
plt.axvline(x=optimal_C_l2, color='r', linestyle='--', label=f'Optimal C = {optimal_C_l2:.4f}')
plt.scatter(optimal_C_l2, optimal_coefficient_l2, color='r', zorder=5, label=f'Optimal Coefficient = {optimal_coefficient_l2:.4f}')

plt.legend()
plt.show()


# In[63]:


# Optimal Coefficient Value and Magnitude

print(f"Optimal Coefficient value (L2 regularization): {optimal_C_l2:.2f}")
print(f"Optimal Coefficient Magnitude: {optimal_coefficient_l2:.2f}")


# In[64]:


# Set up 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform Cross-Validation
cv_scores = cross_val_score(logReg_l2, X_train_selected, y_train_sampled, cv=skf, scoring='accuracy')

# Print results
print(f'Cross-validation Accuracy Scores: {cv_scores}')
print(f'Mean Accuracy: {cv_scores.mean():.2f}')
print(f'Standard Deviation: {cv_scores.std():.4f}')


# In[65]:


# Tune hyperparameter on Logistic Regression for L2 regularization

logReg_l2 = LogisticRegression(penalty='l2', C=optimal_C_l2, l1_ratio = optimal_coefficient_l2, solver='lbfgs', max_iter=500, random_state = 42, class_weight='balanced')
logReg_l2.fit(X_train_selected, y_train_sampled)


# In[66]:


# predict the X_test

y_predd_l2 = logReg_l2.predict(X_test_selected)


# In[67]:


# Track accuracy and coefficients
accuracies_l2 = []

A_score_l2 = accuracy_score(y_test, y_predd_l2)
accuracies_l2.append(A_score_l2)
coefficients_l2.append(np.abs(logReg_l2.coef_[0]))

# Accuracy of the L2 Logistic Regression model

print(f'Accuracy of the L2 Logistic Regression model: {max(accuracies_l2) * 100:.0f}%')


# In[68]:


# ROC curve score L2 Logistic Regression model

roc_auc_score(y_test,y_predd_l2)


# In[69]:


# Visualize ROC curve for L2 Logistic Regression model

plt.figure(figsize=(8,6))

fpr, tpr, thresholds = roc_curve(y_test, y_predd_l2)
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.1])
plt.ylim([0.0, 1.1])
plt.title('ROC curve for Glaucoma disease classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# In[70]:


# Classification report for logistic regression

c_report2 = classification_report(y_test, y_predd_l2)
print(c_report2)


# In[71]:


# Build confusion matrix for Logistic Regression model

CM2 = confusion_matrix(y_predd, y_test)

plt.figure(figsize = (4, 3))
sns.heatmap(CM2, annot = True, fmt = 'g')
plt.title('Confusion Matrix for Logistic Regression model')
plt.xlabel('Predicted')
plt.ylabel('Actual')


# In[72]:


# Define models
models = {
    "RandomForest": RandomForestClassifier(max_depth=10, n_estimators=500, class_weight='balanced', random_state=42),
    "LogisticRegression": LogisticRegression(penalty='l2', C=optimal_C_l2, l1_ratio=optimal_coefficient_l2,
                                             solver='lbfgs', max_iter=500, random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(max_depth=10, n_estimators=500, random_state=42, learning_rate=0.001, eval_metric='logloss'),
    "SVM": SVC(probability=True, max_iter=500, class_weight='balanced', kernel='rbf', random_state=42)
}

results = []
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()  # Flatten to 1D for easy indexing

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train_selected, y_train_sampled)
    train_accuracy = model.predict(X_train_selected)
    y_pred = model.predict(X_test_selected)
    y_score = model.predict_proba(X_test_selected)[:, 1]  # For ROC curve

    # Metrics
    acc_1 = accuracy_score(y_train_sampled, train_accuracy)
    acc_2 = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    RMSE = root_mean_squared_error(y_test, y_pred)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    axes[i].plot(fpr, tpr)
    axes[i].set_xlim([0.0, 1.0])
    axes[i].set_ylim([0.0, 1.0])
    axes[i].set_title(f'ROC Curve ({name})')
    axes[i].set_xlabel('False Positive Rate')
    axes[i].set_ylabel('True Positive Rate')
    axes[i].grid(True)

    # Save metrics
    results.append({
        'Model': name,
        'Train Accuracy': acc_1,
        'Test Accuracy': acc_2,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'RSME': RMSE
    })

    print(f"\nClassification Report for {name}:\n")
    print(classification_report(y_test, y_pred))

plt.tight_layout()
plt.show()

# Create a DataFrame
results_df = pd.DataFrame(results)

# Prepare data for plotting
results_long = results_df.melt(id_vars='Model', value_vars=['Train Accuracy', 'Test Accuracy'],
                               var_name='Dataset', value_name='Accuracy')

# Plot Train vs Test Accuracy
plt.figure(figsize=(8,5))
sns.barplot(x='Model', y='Accuracy', hue='Dataset', data=results_long, palette='Paired')
plt.ylim(0, 1)
plt.title('Train vs Test Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.legend()
plt.show()


# ### Show comparison table

# In[73]:


comparison_df = pd.DataFrame(results).set_index('Model')
comparison_df = comparison_df.drop(columns=['Train Accuracy'])

print("\nModel Performance Comparison:\n")
print(comparison_df.round(3))


# ### Comparison Between RandomForest Model (Chosen Model) And Other Tested Models

# In comparing the performance of Random Forest against other models such as Logistic Regression (L2), XGBoost, and Support Vector Machine (SVM), we found that Random Forest consistently outperformed the others in terms of accuracy.
# 
# With feature selection and fine-tuned hyperparameters—specifically, a maximum depth of 10 and 500 estimators—the Random Forest model achieved an accuracy of 91%. In contrast:
# 
# ##### 1. Logistic Regression (L2) achieved 52% accuracy
# ##### 2. XGBoost reached 79% accuracy
# ##### 3. SVM yielded the lowest accuracy at 48%
# 
# While Random Forest performed best in this configuration, XGBoost has the potential to surpass it depending on how its learning rate is tuned. However, caution is needed when increasing the learning rate:
# 
# A high learning rate can lead to overfitting, where the model fits the training data too closely and fails to generalize.
# To mitigate this, a lower learning rate range (0.001 to 0.1) was used during testing, which provided better generalization and helped prevent overfitting.
# 
# This significant difference suggests that Random Forest is a better choice for this dataset, as it captures more complex patterns and relationships compared to Logistic Regression, which may struggle with non-linear data. The improved performance of Random Forest highlights the importance of hyperparameter tuning and choosing the right model for the problem at hand.

# ### Glaucoma Case Detection

# The model demonstrates high precision of 0.96 for detecting Glaucoma, meaning that when it predicts a patient has Glaucoma, it is correct 96% of the time. This indicates that the model produces fewer false positives, minimizing the chances of incorrectly diagnosing a non-glaucoma case as Glaucoma.
# 
# However, the recall for Glaucoma is 0.86, which means the model correctly identifies 86% of actual Glaucoma cases, but misses the remaining 14% (false negatives). This suggests that some patients with Glaucoma may not be detected by the model.

# ### Non Glaucoma Case Detection

# The model's precision of 0.87 for non Glaucoma cases compared to Glaucoma cases indicates that it sometimes misclassifies non-glaucoma cases as glaucoma. In other words, when the model predicts non Glaucoma, it is only correct 87% of the time, meaning there are some false positives for this class.
# 
# On the other hand, the higher recall of 0.96 for non Glaucoma cases compared to Glaucoma cases shows that the model is effective at identifying most actual non-glaucoma cases. This means that 96% of all true non Glaucoma cases are correctly classified, but some actual non Glaucoma cases about 4% may still be misclassified.

# ### Final Summary for RandomForest Model

# The top 10 most important features predicted by the model for detecting early signs of glaucoma, listed in descending order of importance, are:
# 
# 1. OCT Results (GCC Thickness): Measures the thickness of certain layers in the retina. Thinner layers can indicate early glaucoma.
# 
# 2. Macular Thickness: Checks the thickness of the central part of the retina. Thinning of the macula can indicate the loss of retinal ganglion cells, which is associated with glaucoma.
#   
# 3. Pachymetry: Measures how thick the cornea is. Thinner corneas can mean a higher risk of glaucoma.
# 
# 4. Intraocular Pressure (IOP): Measures the pressure inside the eye. High pressure can damage the optic nerve and lead to glaucoma.
# 
# 5. OCT Results (RNFL Thickness): Measures the thickness of the retinal nerve fiber layer. Thinning can be a sign of glaucoma.
# 
# 6. Retinal Volume: Measures the total volume of the retina. Reduction in retinal volume can indicate the loss of retinal ganglion cells, which is associated with glaucoma.
# 
# 7. Age: Older age increases the risk of developing glaucoma.
# 
# 8. Visual Field Test Results (Sensitivity): This test measures how well you can see in different areas of your vision. Low sensitivity in certain areas of the visual field can indicate glaucoma, as it reflects areas where vision has been lost.
# 
# 9. Cup-to-Disc Ratio (CDR): Compares the size of the optic cup to the optic disc. A larger ratio can indicate glaucoma damage.
# 
# 10. Visual Field Test Results (Specificity): Measures how accurately the test identifies people without glaucoma. Low specificity can lead to false positives, where people without glaucoma are incorrectly identified as having the disease. However, it is not the specificity itself that causes glaucoma, but rather the accuracy of the test in correctly identifying those who do not have the condition.

# The Random Forest model achieves 91% accuracy, demonstrating strong overall performance. It is more effective at detecting Glaucoma than No Glaucoma cases.
# 
# Overall, while the model is highly accurate, it prioritizes precision for Glaucoma but may miss some true cases. Meanwhile, it is better at identifying No Glaucoma cases but occasionally misclassifies Glaucoma as No Glaucoma.

# In[ ]:




