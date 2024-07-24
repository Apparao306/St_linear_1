import streamlit as st
import sklearn
from utils import PrepProcesor, columns 

import numpy as np
import pandas as pd
import joblib

model = joblib.load('lim.joblib')
st.title('Find Salary of Employee')
# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked


Exp = st.number_input("Enter experience of employee", 1,40)
df1 = pd. DataFrame({'YearsExperience': Exp}, index=[0])


def predict(): 
    #X = pd.DataFrame([row], columns = columns)
    prediction = model.predict(df1)
    st.write('Salary of Employee:', prediction)
   

trigger = st.button('Predict', on_click=predict)

