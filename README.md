🚢 Titanic Survival Prediction using Machine Learning

📌 Problem Statement

The Titanic disaster is one of the most well-known shipwrecks in history. This project aims to predict whether a passenger survived or not based on features like age, gender, ticket class, fare, etc.

It is a binary classification problem solved using Machine Learning.

---
TEAM MEMBERS:

Lohar Vikram Gautamkumar (team lead)

Sahil Bhandare 

Rishu Rajkumar Singh

R Likhith Sai

bonam Sai satya thansush

Siba sankar sahu

Shubham Gupta

---

🎯 Project Objective

- Build a predictive model using Logistic Regression
- Perform data preprocessing and feature engineering
- Evaluate model performance
- Generate predictions for unseen data
- Save the trained model for future use

---

📂 Dataset Information

The dataset contains passenger details such as:

- PassengerId
- Pclass (Ticket Class)
- Name
- Sex
- Age
- SibSp (Siblings/Spouses aboard)
- Parch (Parents/Children aboard)
- Ticket
- Fare
- Cabin
- Embarked
- Survived (Target Variable)

---

⚙️ Technologies Used

- Python 🐍
- Pandas
- NumPy
- Scikit-learn
- Pickle

---

🔄 Workflow

1. Data Loading

- Load training and test datasets using Pandas

2. Data Preprocessing

- Dropped unnecessary columns:
  - PassengerId, Name, Ticket, Cabin
- Handled missing values:
  - Age → Median
  - Embarked → Mode
  - Fare → Median

3. Feature Encoding

- Converted categorical variables using One-Hot Encoding
- Aligned test dataset columns with training dataset

4. Feature Selection

- "X" → Input Features
- "y" → Target (Survived)

5. Train-Test Split

- 80% Training Data
- 20% Testing Data

6. Feature Scaling

- Standardized features using StandardScaler

7. Model Training

- Logistic Regression model trained on scaled data

8. Model Evaluation

- Accuracy Score
- Confusion Matrix
- Classification Report

9. Prediction

- Predictions generated for test dataset

10. Model Saving

- Model saved using pickle
- Scaler saved for future transformations

---

📊 Model Performance

- Algorithm Used: Logistic Regression
- Accuracy: ~80% (may vary slightly)

Key Insights:

- Females had higher survival chances
- 1st class passengers were more likely to survive
- Fare and age also influenced survival

---

📁 Project Structure

📦 Titanic-Survival-Prediction
 ┣ 📜 Titanic_train.csv
 ┣ 📜 Titanic_test.csv
 ┣ 📜 titanic_predictions.csv
 ┣ 📜 titanic_model.pkl
 ┣ 📜 scaler.pkl
 ┣ 📜 main.py
 ┗ 📜 README.md
