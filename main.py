import pandas as pd
import numpy as np  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# ==============================
# Step 1: Load Dataset
# ==============================
train_df = pd.read_csv("Titanic_train.csv")
test_df = pd.read_csv("Titanic_test.csv")

# ==============================
# Step 2: Data Preprocessing
# ==============================

# Drop unnecessary columns
train_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
test_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# Fill missing values using NumPy
train_df["Age"] = train_df["Age"].fillna(np.median(train_df["Age"]))
test_df["Age"] = test_df["Age"].fillna(np.median(test_df["Age"]))

train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])
test_df["Embarked"] = test_df["Embarked"].fillna(test_df["Embarked"].mode()[0])

test_df["Fare"] = test_df["Fare"].fillna(np.median(test_df["Fare"]))

# ================================
# Step 3 : Encoding
# ================================
train_df = pd.get_dummies(train_pdf,drop_first=True)
test_df = pd.get_dummies(test_df,drop_first=True)

#Align columns
test_df = test_df.reindex(columns = train_df.drop("Survived",axis=1).columns, fill_value=0)

# ================================
# step 4: Define X and Y
# ================================
X = train_df.drop("Survived",axis=1)
Y = train_df["Survived"]

# Convert to Numpy arrays(explict use)
X = np.array(X)
Y = np.array(Y)

# ==============================
# Step 5: Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Step 6: Scaling
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)