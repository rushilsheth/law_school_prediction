# imports

import boto3
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import streamlit as st
import plotly.express as px
from imblearn.over_sampling import SMOTENC
import operator
import io


st.title("Law School Predictor")

@st.cache()
def load_data(): 
    s3 = boto3.client('s3') # comment out when on local
    obj = s3.get_object(Bucket = 'lawyer-predict', Key = 'df.csv')
    data_fnc = pd.read_csv(io.BytesIO(obj['Body'].read()))

    s3_bucket_stuff = boto3.resource('s3')
    bucket = s3_bucket_stuff.Bucket('lawyer-predict')    
    for file in bucket.objects.all():    
        file_date = file.last_modified.replace(tzinfo=None)
    
    return data_fnc, file_date

data_orig, date_modified = load_data()

date_month_dict = {1: 'January ', 2:'February ', 3:'March ', 4:'April ', 5:'May ', 6:'June ', 7:'July ', 8:'August ', 9:'September ', 10:' October', 11:'November ', 12: 'December '}
month_name = date_month_dict[date_modified.month]

st.markdown(f'Last updated {date_month_dict[date_modified.month]}{date_modified.day}')
st.markdown(
"""
Find out your chances at law school.
I was inspired to make this application while my friend was applying to law school this past year.
She found it very helpful and I hope it can help you too. Please input your true statistics, to get an accurate
probability of your initial decision. Best of luck on your law school apps!

Note: This data dates back to admissions for the 2014-2015 school year and accounts for the most recent trends in admissions
""")

data = data_orig.copy()

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

#SMOTE stuff

# for app
school_orig = data.loc[data.school == school_name][['decision_numeric', 'gpa', 'lsat', 'urm', 'work_experience_encode', 'cycleid']].dropna().reset_index().drop('index', axis =1)

smote_cycle_y = school_orig['cycleid']#.to_numpy()
smote_cycle_x = school_orig.drop('cycleid', axis = 1)#.to_numpy()

samp_dict = {}

for cycle_id_smote in np.unique(smote_cycle_y):
    samp_dict[cycle_id_smote] = smote_cycle_y[smote_cycle_y==cycle_id_smote].shape[0]
samp_dict[np.max(smote_cycle_y)] = int(np.rint(samp_dict[np.max(smote_cycle_y)]*2.1))
oversample = SMOTENC(sampling_strategy = samp_dict, categorical_features = [0,3,4], random_state = 7)

X_up, y_up = oversample.fit_resample(smote_cycle_x,smote_cycle_y)
df_resample_train = pd.DataFrame(X_up,columns = smote_cycle_x.columns)

smote_2_y = df_resample_train[['decision_numeric']]
smote_2_X = df_resample_train.drop('decision_numeric', axis =1)

samp_dict_2 = {}
for cycle_id_smote in np.unique(smote_2_y):
    samp_dict_2[cycle_id_smote] = np.max(smote_2_y['decision_numeric'].value_counts())
oversample_2 = SMOTENC(sampling_strategy = samp_dict_2, categorical_features = [2,3], random_state = 7)

smote_2_y = smote_2_y.decision_numeric

X_fin, y_fin = oversample_2.fit_resample(smote_2_X,smote_2_y)
X_fin = X_fin.to_numpy() #pd.DataFrame(X_fin, columns = smote_2_X.columns)

xgb_school = XGBClassifier(num_classes = 3)
# preds are gpa, lsat, urm, work_experience, 
xgb_school.fit(X_fin, y_fin)


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


# lets treat 
school_orig.group



