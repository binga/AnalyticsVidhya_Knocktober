import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('../input/Train.csv')
test = pd.read_csv('../input/Test.csv')
first_camp = pd.read_csv('../input/First_Health_Camp_Attended.csv',
                         usecols=['Patient_ID', 'Health_Camp_ID', 'Donation', 'Health_Score'])
second_camp = pd.read_csv('../input/Second_Health_Camp_Attended.csv')
third_camp = pd.read_csv('../input/Third_Health_Camp_Attended.csv')
healthcamp_details = pd.read_csv('../input/Health_Camp_Detail.csv')
patient_details = pd.read_csv('../input/Patient_Profile.csv', na_values=["None"])
patient_details = patient_details.fillna(-1)


first_camp['Number_of_stall_visited'] = 1
second_camp['Number_of_stall_visited'] = 1
first_camp['Last_Stall_Visited_Number'] = -1
second_camp['Last_Stall_Visited_Number'] = -1
second_camp['Donation'] = 0
third_camp['Donation'] = 0
second_camp = second_camp.rename(columns={'Health Score': 'Health_Score'})
third_camp['Health_Score'] = 0
third_camp = third_camp[third_camp.Number_of_stall_visited != 0]
all_camps = pd.concat([first_camp, second_camp, third_camp], ignore_index=True)


train['key'] = train.Health_Camp_ID.astype(str) + '_' + train.Patient_ID.astype(str)
all_camps['key'] = all_camps.Health_Camp_ID.astype(str) + '_' + all_camps.Patient_ID.astype(str)
train['Outcome'] = 0
train.loc[train['key'].isin(all_camps['key']), 'Outcome'] = 1
train = train.drop('key', axis=1)


patient_camp_visit_details = all_camps.pivot_table(['Donation', 'Health_Score', 'Number_of_stall_visited'], ['Patient_ID'], aggfunc={'Donation': 'sum','Health_Score' :'mean', 'Number_of_stall_visited':'sum'})
patient_camp_visit_details = patient_camp_visit_details.reset_index()
patient_camp_visit_details.head()


def preprocess(data):
    data = pd.merge(data, healthcamp_details, on='Health_Camp_ID', how='left')
    data = pd.merge(data, patient_details, on='Patient_ID', how='left')
    data['City_Type'] = data['City_Type'].fillna('Z')
    data['Registration_Date'] = pd.to_datetime(data['Registration_Date'])
    data['Camp_Start_Date'] = pd.to_datetime(data['Camp_Start_Date'])
    data.loc[data.Registration_Date.isnull(), 'Registration_Date'] = data.loc[data.Registration_Date.isnull(), 'Camp_Start_Date']
    data['Camp_End_Date'] = pd.to_datetime(data['Camp_End_Date'])
    data['First_Interaction'] = pd.to_datetime(data['First_Interaction'])
    data['Patient_contact_time'] = [int(i.days) for i in (data['Registration_Date'] - data['First_Interaction'])]
    data['Camp_duration'] = [int(i.days) for i in (data['Camp_End_Date'] - data['Camp_Start_Date'])]
    data['Camp_Start_year'] = data.Camp_Start_Date.dt.year.astype(int)
    data['Camp_Start_month'] = data.Camp_Start_Date.dt.month.astype(int)
    return data

train = preprocess(train)
test = preprocess(test)

train = pd.merge(train, patient_camp_visit_details, how='left', on='Patient_ID').fillna(0)
test = pd.merge(test, patient_camp_visit_details, how='left', on='Patient_ID').fillna(0)

train['Days_after_start'] = (train['Registration_Date'] - train['Camp_Start_Date']).dt.days
test['Days_after_start'] = (test['Registration_Date'] - test['Camp_Start_Date']).dt.days

train['camp_awareness'] = (train['First_Interaction'] - train['Camp_Start_Date']).dt.days
test['camp_awareness'] = (test['First_Interaction'] - test['Camp_Start_Date']).dt.days

train['cat1_cat2'] = train['Category1'].astype(str) + '_' + train['Category2'].astype(str)
test['cat1_cat2'] = test['Category1'].astype(str) + '_' + test['Category2'].astype(str)

predictors = ['Income', 'Age', 'City_Type', 'Employer_Category', 'Donation', 'camp_awareness', 'cat1_cat2', 'Days_after_start']
target = train['Outcome']

for f in predictors:
    if (train[f].dtype=='object') or (train[f].dtype=='datetime64[ns]'):
        print(f)
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))


params = {'objective': 'binary:logistic',
          'booster': 'gbtree',
          'eval_metric': 'auc',
          'nthread': 4,
          'silent': 1,
          'max_depth': 4,
          'subsample': 0.5,
          "colsample_bytree": 0.7,
          'eta': 0.040460464,
          'verbose_eval': True,
          'seed': 0}


dtrain = xgb.DMatrix(train[predictors], label=target)
dtest = xgb.DMatrix(test[predictors])

# num_rounds = 224
watchlist = [(dtrain, 'dtrain')]
# clf_xgb_main = xgb.train(dtrain=dtrain, params=params, num_boost_round=num_rounds, evals=watchlist, verbose_eval=10)

num_rounds = 224
preds = np.zeros(len(test))
dtrain = xgb.DMatrix(train[predictors], label=target)
dtest = xgb.DMatrix(test[predictors])
for s in np.random.randint(0, 1000000, size=50):
    params['seed'] = s
    clf_xgb_main = xgb.train(dtrain=dtrain, params=params, num_boost_round=num_rounds, evals=watchlist,
                        verbose_eval=False)
    preds += clf_xgb_main.predict(dtest)
preds = preds/50


submission = test[['Patient_ID', 'Health_Camp_ID']]
submission['Outcome'] = preds
sub_file = '../submissions/submission-knocktober_final.csv'
submission.to_csv(sub_file, index=False)
