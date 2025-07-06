#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install seaborn')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


# In[3]:


url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()


# In[4]:


df.count()


# In[5]:


url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()


# In[6]:


df.count()


# In[7]:


df = df.dropna()
df.info()


# In[8]:


df.columns


# In[9]:


df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })


# In[10]:


df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
df. info()


# In[11]:


def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'


# In[12]:


df['Date'] = pd.to_datetime(df['Date'])        


df['Season'] = df['Date'].apply(date_to_season)


df = df.drop(columns='Date')                  

df


# In[13]:


# Features (all columns except the target)
X = df.drop(columns='RainTomorrow', axis=1)

# Target
y = df['RainTomorrow']


# In[14]:


X = df.drop(columns='Rainfall', axis=1)
y = df['Rainfall']


# In[15]:


y.value_counts()


# In[16]:


y.value_counts()
No     23_814
Yes     6_206
Name: RainTomorrow, dtype: int64


# In[17]:


# Show class counts
counts = y.value_counts()   # pandas Series
print(counts)


# In[18]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,                       # features
    y,                       # target
    test_size=0.20,          # 20 % for testing
    stratify=y,              # keep class ratio the same
    random_state=42          # reproducible split
)


# In[20]:


if y.value_counts().min() >= 2:          # safe to stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
else:                                    # else split without stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )


# In[21]:


numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()


# In[22]:


# Scale the numeric features
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# One-hot encode the categoricals 
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])


# In[23]:


from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# In[24]:


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# In[25]:


param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}


# In[26]:


cv = StratifiedKFold(n_splits=5, shuffle=True)


# In[27]:


from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=pipeline,        # the full preprocessing + model pipeline
    param_grid=param_grid,     # your hyper-parameter dictionary
    cv=5,                      # 5-fold cross-validation
    scoring='accuracy',
    verbose=2,
    n_jobs=-1                  # use all cores (optional)
)

grid_search.fit(X_train, y_train)


# In[28]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────────────────
# 1 ▸ Target & features
# ──────────────────────────────────────────────────────────
# Binary target: 1 if any rain; 0 otherwise
y = (df['Rainfall'] > 0).astype(int)          # <-- or use RainTomorrow column

X = df.drop(columns=['Rainfall'])             # drop target from features

# ──────────────────────────────────────────────────────────
# 2 ▸ Train / test split (with stratification)
# ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ──────────────────────────────────────────────────────────
# 3 ▸ Auto-detect numeric vs. categorical columns
# ──────────────────────────────────────────────────────────
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# ──────────────────────────────────────────────────────────
# 4 ▸ Preprocessing pipelines
# ──────────────────────────────────────────────────────────
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler' , StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot' , OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ──────────────────────────────────────────────────────────
# 5 ▸ Full model pipeline
# ──────────────────────────────────────────────────────────
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier'  , RandomForestClassifier(random_state=42))
])

# ──────────────────────────────────────────────────────────
# 6 ▸ Hyper-parameter grid
# ──────────────────────────────────────────────────────────
param_grid = {
    'classifier__n_estimators'     : [100, 200],
    'classifier__max_depth'        : [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf' : [1, 2]
}

# ──────────────────────────────────────────────────────────
# 7 ▸ Grid search
# ──────────────────────────────────────────────────────────
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best CV accuracy :", grid_search.best_score_)
print("Best parameters  :", grid_search.best_params_)

# ──────────────────────────────────────────────────────────
# 8 ▸ Evaluate on hold-out test set
# ──────────────────────────────────────────────────────────
y_pred = grid_search.predict(X_test)
print("\nTest-set report")
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


# In[29]:


test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))


# In[30]:


y_pred = grid_search.predict(X_test)



# In[31]:


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[32]:


feature_importances = grid_search.best_estimator_['classifier'].feature_importances_


# In[33]:


# Combine numeric and categorical feature names
feature_names = numeric_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

N = 20  # Change this number to display more or fewer features
top_features = importance_df.head(N)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
plt.xlabel('Importance Score')
plt.show()


# In[34]:


from sklearn.linear_model import LogisticRegression

# ── 1 ▸ swap the classifier inside the existing pipeline ─────────────
pipeline.set_params(classifier=LogisticRegression(random_state=42, max_iter=1000))

# ── 2 ▸ point the existing GridSearchCV object at the new pipeline ──
grid_search.estimator = pipeline      # new “model” inside the search

# ── 3 ▸ Logistic-Regression hyper-parameter grid ─────────────────────
param_grid = {
    'classifier__solver'      : ['liblinear'],
    'classifier__penalty'     : ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}
grid_search.param_grid = param_grid   # replace the old forest grid

# ── 4 ▸ fit and predict with the updated search object ───────────────
model = grid_search                    # alias for clarity (optional)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# In[35]:


print(classification_report(y_test, y_pred))

# Generate the confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:




