print("Titanic App is starting...")

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ==============================
# Load Dataset & Train Model
# ==============================
df = pd.read_csv("Titanic_train.csv")

# Drop unnecessary columns
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# Handle missing values
df["Age"] = df["Age"].fillna(np.median(df["Age"]))
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encoding
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==============================
# Routes
# ==============================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        pclass = int(request.form["pclass"])
        age = float(request.form["age"])
        sibsp = int(request.form["sibsp"])
        parch = int(request.form["parch"])
        fare = float(request.form["fare"])
        sex = request.form["sex"]
        embarked = request.form["embarked"]

        # Encoding
        sex_male = 1 if sex == "male" else 0
        embarked_Q = 1 if embarked == "Q" else 0
        embarked_S = 1 if embarked == "S" else 0

        # Input format (IMPORTANT ORDER)
        input_data = np.array([[pclass, age, sibsp, parch, fare,
                                sex_male, embarked_Q, embarked_S]])

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        result = "Survived" if prediction == 1 else "Not Survived"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction="Error: " + str(e))


# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    app.run(debug=True)