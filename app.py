from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

X,y = datasets.load_diabetes(return_X_y=True)
X = X[:, np.newaxis, 3]

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=22)

model = linear_model.LinearRegression()
model.fit(X_train,y_train)

pred= model.predict(X_test)

import matplotlib.pyplot as plt
# The coefficients
print("Coefficients: \n", model.coef_)
# The mean squared error
from sklearn.metrics import mean_squared_error,r2_score
print("Mean squared error: %.2f" % mean_squared_error(y_test, pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, pred))


# Plot outputs
import streamlit as st
number = st.number_input('Insert a number')
number = model.predict([[number]])
st.write('The predict number is %.3f'%number)

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2,2)
ax[0,0].scatter(X_test,y_test)
ax[0,0].plot(X_test,pred)

st.pyplot(fig)

text_contents = '''This is some text'''
st.download_button('Download some text', text_contents)