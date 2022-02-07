"""# Importing Libraries and Data"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

# Importing Libraries
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
from User import User

# For pre-processing data and evaluation results
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# For machine learning
import lightgbm as lgb

# Importing Data 
application = pd.read_csv('data/application_record.csv') # Application records - features
credit = pd.read_csv('data/credit_record.csv') # Credit records - labels

application.iloc[:,:].agg(lambda x:';'.join(str(y) for y in x.unique())).to_frame('Unique Values')


"""# Data Cleaning"""

# Labeling users with 'REJECTED' status (0 - Approved, 1 - Rejected) on 'credit'
credit['REJECTED'] = 0 # Adding 'APPROVED' column with everyone approved
conditions = [credit['STATUS'].isin(['2', '3', '4', '5'])] # Conditions: due more than 60 days. Return TRUE/FALSE
credit['REJECTED'] = np.select(conditions, [1], default = 0) # Assign risky users
users = pd.DataFrame(credit.groupby(['ID']).agg({'REJECTED':'max', 'MONTHS_BALANCE':'min'})) # Unique users record with label and 'OPEN_MONTH'
users = users.rename(columns = {'MONTHS_BALANCE':'OPEN_MONTH'})


# Raw Data = Merge Application + Credit data
# Merge two dataframes 'application' and 'open_month' by 'ID'
raw = pd.merge(application, users, how = 'right', on ='ID')
raw.dropna(inplace = True) # Drop credit records with no information - ANY
# data.dropna(subset = ['CODE_GENDER'], inplace = True) # Drop credit records with no information (not matching applications)

raw = raw.astype({'FLAG_WORK_PHONE': 'int64','FLAG_PHONE': 'int64','FLAG_EMAIL': 'int64'}) # Preserve binary features as integer
raw # Applications with matching credit records


raw.iloc[:,:].agg(lambda x:';'.join(str(y) for y in x.unique())).to_frame('Unique Values')

# Copy raw data to 'data' dataframe ready to be cleaned
data = pd.DataFrame(raw['ID'])

"""# Features
---

## Approval Target
"""

data['Rejected']=raw['REJECTED']

"""## Binary Features

### Gender (CODE_GENDER)
"""

map_array = {'F':0, 'M':1}
data['Gender'] = raw['CODE_GENDER'].astype(str).replace(map_array)

"""### Car Ownership (FLAG_OWN_CAR)"""

map_array = {'N':0, 'Y':1}
data['Car'] = raw['FLAG_OWN_CAR'].astype(str).replace(map_array)

"""### Realty Ownership (FLAG_OWN_REALTY)"""

map_array = {'N':0, 'Y':1}
data['Realty'] = raw['FLAG_OWN_REALTY'].astype(str).replace(map_array)

"""### Mobile Phone (FLAG_MOBIL)"""

# data.drop(labels = 'FLAG_MOBIL', axis = 1, inplace = True, errors = 'ignore') #Dropping column

"""### Work Phone (FLAG_WORK_PHONE)"""

map_array = {'N':0, 'Y':1}
data['Work Phone'] = raw['FLAG_WORK_PHONE'].astype(str).replace(map_array)

"""### Phone (FLAG_PHONE)"""

map_array = {'N':0, 'Y':1}
data['Phone'] = raw['FLAG_PHONE'].astype(str).replace(map_array)

"""### Email (FLAG_EMAIL)"""

map_array = {'N':0, 'Y':1}
data['Email'] = raw['FLAG_EMAIL'].astype(str).replace(map_array)

"""---
## Continuous Features

### Number of Children (CNT_CHILDREN)
"""


data['# of Children'] = np.clip(raw['CNT_CHILDREN'], a_min = None, a_max = 2 ) # Clipping at 2 children and above

"""### Annual Income (AMT_INCOME_TOTAL)

The currency is in CNY (1 USD = 0.16 CNY)
"""

data['Annual Income'], bins_income = pd.qcut(raw['AMT_INCOME_TOTAL']/10000, q=3, labels=["low","medium", "high"], retbins = True)

"""### Age (DAYS_BIRTH)"""


data['Age'], bins_age = pd.cut(-raw['DAYS_BIRTH']//365, bins=5, labels=["lowest","low","medium","high","highest"], retbins = True) #qcut-q, cut-bins

"""### Years Employed (DAYS_EMPLOYED)"""


data['Years Employed'], bins_years_employed = pd.cut(-raw['DAYS_EMPLOYED']//365, bins=5, labels=["lowest","low","medium","high","highest"], retbins = True) #qcut-q, cut-bins

"""### Family Size (CNT_FAM_MEMBERS)"""


data['Family Size'] = np.clip(raw['CNT_FAM_MEMBERS'], a_min = None, a_max = 3 ) # Clipping at 3 family size and above
#data['# of Children'].plot(kind='hist', xticks = [0,1,2])

"""---
## Categorical Features

### Income Category (NAME_INCOME_TYPE)
"""


#TODO consider not to/to merge student and pensioner to state servant?
map_array = {'State servant': ['Pensioner', 'Student']}
data['Income Category'] = raw['NAME_INCOME_TYPE']
for new_value in map_array:
  data['Income Category'].replace(map_array[new_value], new_value, inplace=True)

"""### Occupation (OCCUPATION_TYPE)"""


map_array = {'Labor'    : ['Laborers','Drivers','Cooking staff','Security staff','Cleaning staff','Low-skill Laborers','Waiters/barmen staff'],
             'Office'   : ['Core staff','Sales staff','Accountants','Medicine staff','Private service staff','Secretaries','HR staff','Realty agents'],
             'High Tech': ['Managers','High skill tech staff','IT staff']}
data['Occupation'] = raw['OCCUPATION_TYPE']
for new_value in map_array:
  data['Occupation'].replace(map_array[new_value], new_value, inplace=True)

"""### Education (NAME_EDUCATION_TYPE)"""


map_array = {'Higher education' : ['Academic degree']}             
data['Education'] = raw['NAME_EDUCATION_TYPE']
for new_value in map_array:
  data['Education'].replace(map_array[new_value], new_value, inplace=True)

"""### Marital Status (NAME_FAMILY_STATUS)"""


data['Marital Status'] = raw['NAME_FAMILY_STATUS']

"""### Residency (NAME_HOUSING_TYPE)"""


data['Residency'] = raw['NAME_HOUSING_TYPE']

"""# One Hot Encoding"""

data

#One-Hot Encoding (OHE) the following columns:
OHE_columns=['# of Children','Annual Income','Age','Years Employed','Family Size','Income Category','Occupation','Education','Marital Status','Residency']
data_OHE = pd.get_dummies(data, columns=OHE_columns)
data_OHE.drop('ID',axis=1, inplace=True)



f, ax = plt.subplots(figsize=(8, 8))
corr = data_OHE.iloc[:,:].corr()
threshold=np.where(abs(corr) < 0.1, 0, corr)
mask=np.where(threshold==0,True,False)
sns.set(font_scale=0.7)
sns.heatmap(corr, cmap=sns.diverging_palette(10, 130, n=11), vmin=-1,
            square=True, ax=ax, annot=threshold, mask=mask)

"""# Algorithms

## Prepare Training Data
"""
all_columns = list(data_OHE.columns)
label = ['Rejected']
all_features = [feature for feature in all_columns if feature not in label]

X=data_OHE[all_features]
X = X.astype(int)
Y=data_OHE[label]


user = User(all_features)
user.default()
user_df = pd.DataFrame(user.to_dict(all_features), index=[0])


# Splitting test/train data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,
                                                    random_state = 1)

# Balancing imbalanced data
X_train,Y_train = SMOTE(sampling_strategy=0.1).fit_resample(X,Y)
X_train,Y_train = RandomUnderSampler(sampling_strategy=1).fit_resample(X_train,Y_train)


"""## LightGBM"""

lgb_train = lgb.Dataset(X_train, Y_train)
lgb_valid = lgb.Dataset(X_test, Y_test, reference = lgb_train)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate':0.5,        
    'n_estimators':100
}

bst = lgb.LGBMModel(**params)
bst.fit(X_train, Y_train, eval_set=[(X_test, Y_test)],eval_metric='auc',
        eval_names='Test dataset',callbacks=[lgb.log_evaluation(50)])


Y_pred_prob = bst.predict(X_test)
single_Y_pred_prob = bst.predict(user_df)

plt.scatter(range(len(Y_pred_prob)),np.sort(Y_pred_prob))
plt.show()

print('Single test prediction: ', 'Approved' if single_Y_pred_prob<0.5 else 'Rejected')
print('Predicted : ',single_Y_pred_prob[0])
print('Estimated credit score = ',(1-single_Y_pred_prob[0])*300+400)

Y_pred = [0 if x<0.5 else 1 for x in Y_pred_prob]
cm = confusion_matrix(Y_test,Y_pred)

print('Accuracy Score is {:.5}'.format(accuracy_score(Y_test, Y_pred)))
print(pd.DataFrame(cm))
print(classification_report(Y_test, Y_pred))

ax = sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

ax.set_title('Confusion Matrix (LightGBM)')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')

ax.xaxis.set_ticklabels(['Approved','Rejected'])
ax.yaxis.set_ticklabels(['Approved','Rejected'])

## Display the visualization of the Confusion Matrix.
plt.show()

