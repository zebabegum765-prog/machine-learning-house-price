import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("dataset.csv")

X = data[['Size', 'Bedrooms']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict([[1100, 2]])

print("Predicted Price:", prediction[0])
print("Accuracy:", model.score(X_test, y_test))
