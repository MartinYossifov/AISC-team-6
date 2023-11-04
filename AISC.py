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

columns_to_scale = ['age', 'balance', 'duration', 'campaign']

scaler = StandardScaler()

df_encoded[columns_to_scale] = scaler.fit_transform(df_encoded[columns_to_scale])
#%%
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df[columns_to_scale].corr()

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

categorical_cols = df.select_dtypes(include=['object']).columns.drop('y')

#compute Cramer's V for each pair of categorical variables
cramers_v_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)
for col1 in categorical_cols:
    for col2 in categorical_cols:
        cramers_v_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

plt.figure(figsize=(12, 8))
sns.heatmap(cramers_v_matrix.astype(float), annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title('Cramér\'s V Heatmap for Categorical Variables')
plt.show()

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

columns_to_scale = ['age', 'balance', 'duration', 'campaign']

scaler = StandardScaler()

df_encoded[columns_to_scale] = scaler.fit_transform(df_encoded[columns_to_scale])
#%%
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df[columns_to_scale].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()
#%%
from scipy.stats import chi2_contingency
import numpy as np

#https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorial-categorial association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

categorical_cols = df.select_dtypes(include=['object']).columns.drop('y')

#compute Cramer's V for each pair of categorical variables
cramers_v_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)
for col1 in categorical_cols:
    for col2 in categorical_cols:
        cramers_v_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

plt.figure(figsize=(12, 8))
sns.heatmap(cramers_v_matrix.astype(float), annot=True, cmap='coolwarm', vmin=0, vmax=1)
plt.title('Cramér\'s V Heatmap for Categorical Variables')
plt.show()