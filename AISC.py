import pandas as pd

with open(r'C:\Users\Martin Yossifov\OneDrive\Documents\ECS 171\bank-full.csv', 'r') as f:
    lines = f.readlines()
    data = [line.strip().split(';') for line in lines]

df = pd.DataFrame(data[1:], columns=data[0]).applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)

df.columns = [col.strip('"') for col in df.columns]
#df = df.drop(columns=['month', 'day', 'previous', 'pdays'])

#%%
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
#%%
print(df.describe())

for col in df.select_dtypes(include=['object']).columns:
    print(df[col].value_counts())
    print("\n")
#%%
columns_to_convert = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(df.dtypes)

#%%
import matplotlib.pyplot as plt
import seaborn as sns

df.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()

categorical_cols = df.select_dtypes(include=['object']).columns.drop('y')
for col in categorical_cols:
    sns.countplot(data=df, x=col)
    plt.xticks(rotation=45)
    plt.show()
#%%
sns.countplot(data=df, x='y')
plt.show()

#%%
for col in categorical_cols:
    sns.countplot(data=df, x=col, hue='y')
    plt.xticks(rotation=45)
    plt.show()

numerical_cols = df.select_dtypes(exclude=['object']).columns
for col in numerical_cols:
    sns.boxplot(data=df, x='y', y=col)
    plt.show()
#%%
# Drop the specified columns right before one-hot encoding
df = df.drop(columns=['month', 'day', 'previous', 'pdays'])
#%%
#one-hot encoding categorical columns
df_encoded = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns.drop('y'))
#%%
from sklearn.preprocessing import StandardScaler

# Columns to scale
columns_to_scale = ['age', 'balance', 'duration', 'campaign']

# Initialize the scaler
scaler = StandardScaler()

# Apply scaling to the columns
df_encoded[columns_to_scale] = scaler.fit_transform(df_encoded[columns_to_scale])
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix
correlation_matrix = df[columns_to_scale].corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()
#%%
from scipy.stats import chi2_contingency
import numpy as np

def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorial-categorial association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

# List of categorical columns (excluding target 'y')
categorical_cols = df.select_dtypes(include=['object']).columns.drop('y')

#compute Cramer's V for each pair of categorical variables
cramers_v_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)
for col1 in categorical_cols:
    for col2 in categorical_cols:
        cramers_v_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(cramers_v_matrix.astype(float), annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title('Cramér\'s V Heatmap for Categorical Variables')
plt.show()

#%%
from sklearn.model_selection import train_test_split

X = df_encoded.drop('y', axis=1)  # Features (drop the target variable)
y = df_encoded['y']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
from sklearn.linear_model import LogisticRegression

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)


#%%
from sklearn.metrics import classification_report, accuracy_score

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
#%%
from sklearn.ensemble import RandomForestClassifier

# Drop the specified columns
df_rf = df

# Label encode the categorical columns
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in df_rf.select_dtypes(include=['object']).columns:
    df_rf[col] = le.fit_transform(df_rf[col])

# Splitting data into training and test sets
X_rf = df_rf.drop('y', axis=1)  # Features (drop the target variable)
y_rf = df_rf['y']  # Target variable

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=10)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

# Predictions using the Random Forest model
rf_y_pred = rf_model.predict(X_test_rf)

# Evaluation
print(accuracy_score(y_test_rf, rf_y_pred))
print(classification_report(y_test_rf, rf_y_pred))
#%%
# If you haven't installed XGBoost, you need to do so using pip:
# !pip install xgboost

# Import XGBoost Classifier
import xgboost as xgb

# Initialize and train the XGBoost model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=12)
xgb_model.fit(X_train_rf, y_train_rf)

# Predictions using the XGBoost model
xgb_y_pred = xgb_model.predict(X_test_rf)

# Evaluation
print(accuracy_score(y_test_rf, xgb_y_pred))
print(classification_report(y_test_rf, xgb_y_pred))


