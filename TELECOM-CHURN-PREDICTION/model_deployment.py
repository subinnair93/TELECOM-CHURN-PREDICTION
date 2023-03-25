import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib




st.title('Model Deployment:Random Forest')

st.sidebar.header('Enter the details')

def user_input_features():
    account_length=st.sidebar.number_input('account_length')
    voice_plan=st.sidebar.number_input('voice_plan')
    voice_messages=st.sidebar.number_input('voice_messages')
    intl_plan=st.sidebar.number_input('intl_plan')
    intl_mins=st.sidebar.number_input('intl_mins')
    intl_calls=st.sidebar.number_input('intl_calls')
    ints_charge=st.sidebar.number_input('ints_charge')
    days_mins=st.sidebar.number_input('days_mins')
    day_calls=st.sidebar.number_input('day_calls')
    day_charge=st.sidebar.number_input('day_charge')
    eve_mins=st.sidebar.number_input('eve_mins')
    eve_calls=st.sidebar.number_input('eve_calls')
    eve_charge=st.sidebar.number_input('eve_charge')
    night_mins=st.sidebar.number_input('night_mins')
    night_calls=st.sidebar.number_input('night_calls')
    night_charge=st.sidebar.number_input('night_charge')
    customer_calls=st.sidebar.number_input('customer_calls')
    state=st.sidebar.number_input('state')
    area_code=st.sidebar.number_input('area_code')


    
    data={'account_length':account_length,
         'voice_plan':voice_plan,
         'voice_messages':voice_messages,
         'intl_plan':intl_plan,
         'intl_mins':intl_mins,
         'intl_calls':intl_calls,
         'ints_charge':ints_charge,
         'days_mins':days_mins,
         'day_calls':day_calls,
         'day_charge':day_charge,
         'eve_mins':eve_mins,
         'eve_calls':eve_calls,
         'eve_charge':eve_charge,
         'night_mins':night_mins,
         'night_calls':night_calls,
         'night_charge':night_charge,
         'customer_calls':customer_calls,
         'state':state,
         'area_code':area_code}
    
    features=pd.DataFrame(data,index=[0])
    return features


data=user_input_features()
st.subheader('user input parameters')
st.write(data)

model2=joblib.load('model_final.sav')

prediction=model2.predict((data))
prediction_proba=model2.predict_proba((data))

if data.iloc[0,0]:
    st.write('customer is going to churn' if prediction_proba[0][1]>0.5 else 'Customer is not going to CHURN' )


        


