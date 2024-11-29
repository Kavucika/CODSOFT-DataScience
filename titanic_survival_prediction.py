import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data=pd.read_csv('C:/Users/Asus/Desktop/kavucika/Titanic-Dataset.csv')
print(data)
print(data.head())

print(data.isnull().sum())

#dropping irrelevant columns i.e, name, cabin , ticket
data = data.drop(columns=['Name', 'Cabin', 'Ticket'], axis=1)

# Filling missing values in columns age and embarked
data = data.dropna(subset=['Age', 'Embarked'])  
data['Age'] = data['Age'].fillna(data['Age'].median())  
data['Fare'] = data['Fare'].fillna(data['Fare'].mean()) 

# converting categorical variables to numerical format
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})  
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}) 

# Splitting dataset into features and labels
X = data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']] 
y = data['Survived'] 
print(X)
print(y)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#prediction
def predict_survival(age, sex, pclass, fare, embarked):
    user_data = np.array([[pclass, sex, age, fare, embarked]])
    survival_prediction = model.predict(user_data)
    if survival_prediction == 1:
        print("The passenger is predicted to have survived.")
    else:
        print("The passenger is predicted to not have survived.")

age = float(input("Enter the age: "))
sex = int(input("Enter gender (0 for male, 1 for female): "))
pclass = int(input("Enter ticket class (1, 2, or 3): "))
fare = float(input("Enter fare: "))
embarked = int(input("Enter Embarked (0 for C, 1 for Q, 2 for S): "))

predict_survival(age, sex, pclass, fare, embarked)
