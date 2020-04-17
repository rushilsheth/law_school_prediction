# imports
import datetime
from flask import Flask
from flask import render_template, render_template_string, redirect
import simplejson
import urllib.request
import boto3
import time
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import streamlit as st
import plotly.express as px


def read_s3_obj(bucket_name, output_file):
    """ Read from s3 bucket"""
    try:
        s3 = boto3.resource('s3')
        obj = s3.Object(bucket_name, output_file)
        body = obj.get()['Body'].read().decode('utf-8')
        return body
    except:
        return ""


st.title("Law School Predictor")

st.markdown(
"""
Find out your chances at law school.
I was inspired to make this application while my friend was applying to law school this past year.
She found it very helpful and I hope it can help you too. Please input your true statistics, to get an accurate
probability of your initial decision. Best of luck on your law school apps!
""")

data = pd.read_csv('data_full.csv')


#categorize decisions as acceptance as 1 , waitlisted as 2 or denied as 0
data.loc[data.decision.isin(['Accepted', 'Accepted Attending', 'Accepted Deferred', 'Accepted Deferred Attending', 'Accepted Deferred Withdrawn', 'Accepted Withdrawn']), 'decision_numeric'] = 1
data.loc[data.decision.isin(['Rejected', 'Rejected Attending', 'Rejected Deferred', 'Rejected Withdrawn']), 'decision_numeric'] = 0
data.loc[data.decision.isin(['WL Accepted', 'WL Accepted Attending', 'WL Accepted Deferred', 'WL Accepted Deferred Attending', 'WL Accepted Withdrawn', 'WL Rejected', 'WL Rejected Deferred', 'WL Rejected Withdrawn', 'WL Withdrawn', 'Waitlisted', 'Waitlisted Attending', 'Waitlisted Deferred', 'Waitlisted Withdrawn']), 'decision_numeric'] = 2
data.loc[data.decision.isin(['Intend to Apply', 'Intend to Apply Attending', 'Intend to Apply Deferred Withdrawn', 'Intend to Apply Withdrawn', 'Pending', 'Pending Attending', 'Pending Deferred', 'Pending Deferred Withdrawn', 'Pending Withdrawn']), 'decision_numeric'] = np.nan

data = data.loc[~data.decision_numeric.isna()]
#drop where job experience not given
data = data.loc[data.work_experience != '?']
#label encode work experience
data.loc[data.work_experience == '0 years (KJD)', 'work_experience_encode'] = 0
data.loc[data.work_experience == '1-4 years', 'work_experience_encode'] = 1
data.loc[data.work_experience == '5-9 years', 'work_experience_encode'] = 2
data.loc[data.work_experience == '10+ years', 'work_experience_encode'] = 3

vandy_index = list(np.unique((data.school))).index('Vanderbilt University')

##### SIDEBAR ######
school_name = st.sidebar.selectbox("School you are Interested In", np.unique((data.school)), vandy_index)
gpa = st.sidebar.slider('GPA', 0.0,4.0,3.5)
lsat = st.sidebar.slider('LSAT', 0,180,165)
work_experience = st.sidebar.selectbox('work experience', ['0 years (KJD)', '1-4 years', '5-9 years', '10+ years'])
urm = st.sidebar.selectbox('URM', ['No','Yes'])

urm_dict = {'Yes': 1, 'No': 0}
work_experience_dict = {'0 years (KJD)':0, '1-4 years':1, '5-9 years': 2, '10+ years': 3}

school = data.loc[data.school == school_name][['decision_numeric', 'gpa', 'lsat', 'urm', 'work_experience_encode', 'cycleid']].dropna()

# upsample 
df_cycle17 = school.loc[school.cycleid == 17]

school =school.append([df_cycle17]*3,ignore_index=True)

school_y = school['decision_numeric'].to_numpy()
school_X = school.drop(['decision_numeric', 'cycleid'], axis = 1).to_numpy()

xgb_school = XGBClassifier(num_classes = 3)
# preds are gpa, lsat, urm, work_experience, 
xgb_school.fit(school_X, school_y)


prediction = xgb_school.predict_proba(np.array([gpa,lsat,work_experience_dict[work_experience],urm_dict[urm]]).reshape(1,4))
prediction_df = pd.DataFrame(prediction.T)#, ['Rejected', 'Accepted', 'Waitlist'])
prediction_df.columns = ['Probability of Decision']
prediction_df['Probability of Decision'] = 100*np.round(prediction_df['Probability of Decision'],4)
prediction_df['Decision'] = ['Rejected', 'Accepted', 'Waitlist']
#prediction_df = prediction_df.sort_values('Probability of Decision')
prediction_df['order'] = ['Least Likely', 'Second Most Likely', 'Most Likely']

bar_labels = np.round(prediction_df['Probability of Decision'],4).astype(str)
bar_labels = bar_labels + '%'

import plotly.graph_objects as go


fig = go.Figure(data=[go.Bar(
    y=prediction_df.Decision,
    x=prediction_df['Probability of Decision'],
    #color = prediction_df['Decision'],
    marker_color=['crimson', 'blue', 'yellow'], # marker color can be a single color value or an iterable
    text = bar_labels,
    #marker = dict(color = prediction_df['Decision']),
    orientation='h',
    textposition = 'auto',
    hoverinfo = 'skip'
)])
fig.update_layout(title_text='Probability of Each Decision')
st.plotly_chart(fig)





