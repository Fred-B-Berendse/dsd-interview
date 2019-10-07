# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE, RFECV
from datetime import date
from datetime import datetime
import calendar
from sklearn import metrics


#%%
df = pd.read_csv('data/Interview.csv')


#%%
df.head(2)


#%%
df.columns 

#%% [markdown]
# ## Dropping unneccessary columns

#%%
df.drop(['Candidate Native location',          'Name(Cand ID)',          'Unnamed: 23',          'Unnamed: 24',          'Unnamed: 25',          'Unnamed: 26',          'Unnamed: 27'],         axis=1,         inplace=True)
df = df.drop(1233)

#%% [markdown]
# ## Renaming columns

#%%
df.columns = ['date',
              'client',
              'industry',
              'location',
              'position',
              'skillset',
              'interview_type',
              'gender',
              'cand_cur_loc',
              'cand_job_loc',
              'interview_loc',
              'start_perm',
              'unsch_mtgs',
              'precall',
              'alt_num',
              'res_jd',
              'venue_clear',
              'letter_shared',
              'exp_attend',
              'obs_attend',
              'mar_status'] 


#%%
df.head()

#%% [markdown]
# ## Cleaning Date 
# credit to: https://www.kaggle.com/athoul01/predicting-interview-attendence

#%%
def clean_date(date):
    date = date.str.strip()
    date = date.str.split("&").str[0]
    date = date.str.replace('–', '/')
    date = date.str.replace('.', '/')
    date = date.str.replace('Apr', '04')
    date = date.str.replace('-', '/')
    date = date.str.replace(' ', '/')
    date = date.str.replace('//+', '/')
    return date


#%%
df['date'] = clean_date(df['date'])
df.head()


#%%
# create fresh, new date column
df['date'] = clean_date(df['date'])

# change dates to datetime
df['date'] = pd.to_datetime(df['date'])


#%%
# change datetimes to days of week

def make_weekdays_column():
    weekdays = []
    for i in range(0,len(df['date'])):
        weekdays.append(df['date'][i].weekday())
    weekdays = pd.Series(weekdays)
    df['weekdays'] = weekdays
#     print(df['weekdays'])


#%%
make_weekdays_column()
df

#%% [markdown]
# ## Finding distances

#%%
def clean_locations(series):
    # Cleans locations in a pandas series
    ext = series.str.extract('(\w+)')
    ext[0] = ext[0].replace('Gurgaonr','Gurgaon')
    return ext[0].str.capitalize()


#%%
df['location'] = clean_locations(df['location'])
df['cand_cur_loc'] = clean_locations(df['cand_cur_loc'])
df['cand_job_loc'] = clean_locations(df['cand_job_loc'])
df['interview_loc'] = clean_locations(df['interview_loc'])


#%%
def get_distance(series1,series2):
    # Returns the L2 distance (in km) from series1 to series2; 
    # series1 and series2 are pandas series containing locations 
    # From https://distancecalculator.globefeed.com/India_Distance_Calculator.asp   
    loc_list = ['Bangalore', 'Chennai', 'Cochin', 
                'Delhi', 'Gurgaon', 'Hosur', 
                'Hyderabad', 'Noida', 'Visakapatinam']

    distances = np.array(
        [[   0,  290,  354, 1750,  346,  248,   39, 2127,  799],
         [ 290,    0,  692, 1768, 1742,  268,  515, 1748,  602],
         [ 354,  692,    0, 2090, 2062,  356,  863, 2675, 1429],
         [1750, 1768, 2090,    0,   43, 1777, 1266,   46, 1375],
         [ 346, 1742, 2062,   43,    0, 1750, 1240,   51, 1354],
         [ 248,  268,  356, 1777, 1750,    0,  608, 1757,  798],
         [  39,  515,  863, 1266, 1240,  608,    0, 1536,  503],
         [2127, 1748, 2675,   46,   51, 1757, 1536,    0, 1344],
         [ 799,  602, 1429, 1375, 1354,  798,  503, 1344,    0]]
         )

    dist_list = []
    for loc1, loc2 in zip(series1.values, series2.values):
        idx1 = loc_list.index(loc1)
        idx2 = loc_list.index(loc2)
        dist_list.append(distances[idx1,idx2])
    return pd.Series(dist_list, index=series1.index) 


#%%
df['d_loc2job'] = get_distance(df['location'], df['cand_job_loc'])
df['d_loc2int'] = get_distance(df['location'], df['interview_loc'])

#%% [markdown]
# ## Columns to convert to 1s and 0s
# 'Gender'
# 
# 'Have you obtained the necessary permission to start at the required time'
# 
# 'Hope there will be no unscheduled meetings'
# 
# 'Can I Call you three hours before the interview and follow up on your attendance for the interview'
# 
# 'Can I have an alternative number/ desk number. I assure you that I will not trouble you too much'
# 
# 'Have you taken a printout of your updated resume. Have you read the JD and understood the same'
# 
# 'Are you clear with the venue details and the landmark.'
# 
# 'Has the call letter been shared'
# 
# 'Expected Attendance'
# 
# 'Observed Attendance' 
# 
# 'Marital Status'

#%%
def convert_yes_no_to_bool(df, col, str_to_match):
    series = pd.Series(np.where(df[col].values == str_to_match, 1, 0))
    df[col] = series
    return df


#%%
df = convert_yes_no_to_bool(df, 'obs_attend', 'Yes')
df = convert_yes_no_to_bool(df, 'start_perm', 'Yes')
df = convert_yes_no_to_bool(df, 'unsch_mtgs', 'Yes')
df = convert_yes_no_to_bool(df, 'precall', 'Yes')
df = convert_yes_no_to_bool(df, 'alt_num', 'Yes')
df = convert_yes_no_to_bool(df, 'res_jd', 'Yes')
df = convert_yes_no_to_bool(df, 'venue_clear', 'Yes')
df = convert_yes_no_to_bool(df, 'letter_shared', 'Yes')
df = convert_yes_no_to_bool(df, 'exp_attend', 'Yes')
# 1 for married 0 for single
df = convert_yes_no_to_bool(df, 'mar_status', 'Married')
# 1 for male 0 for female
df = convert_yes_no_to_bool(df, 'gender', 'Male')


#%%
df.head(10)

#%% [markdown]
# ## Dropping columns to run models

#%%
df.drop(['date',          'client',          'industry',          'location',          'position',          'skillset',          'interview_type',          'cand_cur_loc',          'cand_job_loc',          'interview_loc'],         axis=1,         inplace=True)
df.head()

#%% [markdown]
# ## Creating train and test 

#%%
y = df.pop('obs_attend').values
X = df.values


#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

#%% [markdown]
# # Creating and running Models
#%% [markdown]
# ## Random Forest Classifier

#%%
rfs = RandomForestClassifier(n_estimators=100, random_state=5, oob_score=True)


#%%
rfs.fit(X_train, y_train)


#%%
print("Training Acc:", round(rfs.score(X_train, y_train),5)),
print("Validation Acc:", round(rfs.score(X_test, y_test), 5))
print("Out-of-Bag Acc:", round(rfs.oob_score_, 5))
print("Score: {}".format(rfs.score(X_test, y_test)))


#%%
probs = rfs.predict_proba(X_test)
probs = [p[1] for p in probs]
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
roc_auc = metrics.roc_auc_score(y_test, probs)


#%%
plt.plot(fpr, tpr, color = 'darkorange',
        label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest with all Categorical Variables and Day of the Week')
plt.savefig('graphs/rf_cats_day.jpg')

#%% [markdown]
# ## Random Forest with min samples leaf set at 5

#%%
rfs = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, oob_score=True, random_state=5)
rfs.fit(X_train, y_train)


#%%
print("Training Acc:", round(rfs.score(X_train, y_train),5)),
print("Validation Acc:", round(rfs.score(X_test, y_test), 5))
print("Out-of-Bag Acc:", round(rfs.oob_score_, 5))
print("Score: {}".format(rfs.score(X_test, y_test)))


#%%
plt.plot(fpr, tpr, color = 'darkorange',
        label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest with min samples at 5')
plt.savefig('graphs/rf_cats_day_minsam-.jpg')

#%% [markdown]
# ## Gradient Boosting Classifier

#%%
gb = GradientBoostingClassifier(loss='exponential', learning_rate=0.05, random_state=10)
gb.fit(X_train, y_train)


#%%
print("Training Acc:", round(gb.score(X_train, y_train),5)),
print("Validation Acc:", round(gb.score(X_test, y_test), 5))


#%%
plt.plot(fpr, tpr, color = 'darkorange',
        label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Gradient Boosting')
plt.savefig('graphs/gb.jpg')

#%% [markdown]
# ## K Nearest Neighbors

#%%
knn = KNeighborsClassifier(n_neighbors=8)


#%%
knn.fit(X_train, y_train)


#%%
knn.score(X_test, y_test)


#%%
probs = knn.predict_proba(X_test)
probs = [p[1] for p in probs]
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
roc_auc = metrics.roc_auc_score(y_test, probs)

plt.plot(fpr, tpr, color = 'darkorange',
        label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN with all Categorical Variables and Day of the Week')
plt.savefig('graphs/knn_cats_day.jpg')

#%% [markdown]
# ## Running models with Feature Selection
#%% [markdown]
# ### Random Forest

#%%
selector = RFE(rfs, 5, step=1)
selector = selector.fit(X_train, y_train)
selector.score(X_test, y_test)
print(f'Feature ranking {selector.ranking_}')

#%% [markdown]
# ### Gradient Boosting

#%%
selector = RFE(gb, 5, step=1)
selector = selector.fit(X_train, y_train)
selector.score(X_test, y_test)
print(f'Feature ranking {selector.ranking_}')

#%% [markdown]
# ## Feature Selection with Cross Validation
#%% [markdown]
# ### Random Forest

#%%
selector = RFECV(rfs, cv=3, step=1)
selector = selector.fit(X_train, y_train)
print(f'Validation Score {selector.score(X_test, y_test)}')
print(f'Feature ranking {selector.ranking_}')

#%% [markdown]
# ### Gradient Boosting

#%%
selector = RFECV(gb, cv=3, step=1)
selector = selector.fit(X_train, y_train)
selector.score(X_test, y_test)
print(f'Validation Score {selector.score(X_test, y_test)}')
print(f'Feature ranking {selector.ranking_}')


#%%
df.columns

#%% [markdown]
# ### Dropping all non categorical columns

#%%
df.drop(['weekdays'], axis=1, inplace=True)
df.head()


#%%
X = df.values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

#%% [markdown]
# # Creating and fitting models
#%% [markdown]
# ## Random Forest Classifier

#%%
rfs = RandomForestClassifier(n_estimators=100, random_state=5)


#%%
rfs.fit(X_train, y_train)


#%%
rfs.score(X_test, y_test)


#%%
probs = rfs.predict_proba(X_test)
probs = [p[1] for p in probs]
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
roc_auc = metrics.roc_auc_score(y_test, probs)

plt.plot(fpr, tpr, color = 'darkorange',
        label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest with only Categorical Variables')
plt.savefig('graphs/rf_cats_only.jpg')


#%%
selector = RFECV(rfs, cv=3, step=1)
selector = selector.fit(X_train, y_train)
print(f'Validation Score {selector.score(X_test, y_test)}')
print(f'Feature ranking {selector.ranking_}')

#%% [markdown]
# ## K Nearest Neighbors

#%%
knn = KNeighborsClassifier(n_neighbors=7)


#%%
knn.fit(X_train, y_train)


#%%
knn.score(X_test, y_test)


#%%
probs = knn.predict_proba(X_test)
probs = [p[1] for p in probs]
fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
roc_auc = metrics.roc_auc_score(y_test, probs)

plt.plot(fpr, tpr, color = 'darkorange',
        label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN with only Categorical Variables')
plt.savefig('graphs/knn_cats_only.jpg')


#%%



#%%



#%%



