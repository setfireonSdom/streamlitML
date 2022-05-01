import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report,confusion_matrix

data=pd.read_excel('./Date_Fruit_Datasets/Date_Fruit_Datasets.xlsx')
x=data.iloc[:,:-1]
le=LabelEncoder()
le.fit(data["Class"])
y=le.transform(data["Class"])

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,stratify=y)

model_type=[LogisticRegression(),DecisionTreeClassifier(),SVC()]


import streamlit as st
import numpy as np
with st.container():
    data
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True)
    fig.set_size_inches(20, 12)
    st.write('This is heatmap')
    st.pyplot(fig)

st.write('start analysis!!!!!!!!!!!!!!!!')
model_li = []
for i,model in enumerate(model_type):
    pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(x_train, y_train)
    y_predict = pipe.predict(x_test)
    model_li.append(pipe)
    with st.container():
        st.write(model, " Classification Report \n",
              classification_report(y_test, y_predict, target_names=data["Class"].unique()))

    cm_data = pd.DataFrame(confusion_matrix(y_test, y_predict), index=data["Class"].unique(),
                           columns=data["Class"].unique())
    new_fig = plt.figure(i+100)
    sns.heatmap(cm_data, annot=True, annot_kws={"size": 18}, fmt="d")
    plt.title(model)
    plt.ylabel('Actual Classes')
    plt.xlabel('Predicted Classes')
    st.pyplot(new_fig)

