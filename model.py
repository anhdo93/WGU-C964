# Importing Libraries
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt

# Importing Data 
application = pd.read_csv('/content/drive/Othercomputers/My Computer/Desktop/application_record.csv') # Application records - features
credit = pd.read_csv('/content/drive/Othercomputers/My Computer/Desktop/credit_record.csv') # Credit records - labels

application.iloc[:,:].agg(lambda x:';'.join(str(y) for y in x.unique())).to_frame('Unique Values')

# Application records
print(application)
application.info()
application.describe()
application.nunique()

# Credit records
print(credit)
credit.info()
credit.describe()
credit.nunique()

"""# Data Cleaning"""

# Labeling users with 'APPROVED' (0 - Not approved, 1 - Approved) on 'credit'
credit['APPROVED'] = 1 # Adding 'APPROVED' column with everyone approved
conditions = [credit['STATUS'].isin(['2', '3', '4', '5'])] # Conditions: due more than 60 days
credit['APPROVED'] = np.select(conditions, [0], default = 1) # Assigned as risky users - not approved
users = pd.DataFrame(credit.groupby(['ID']).agg({'APPROVED':'min', 'MONTHS_BALANCE':'min'})) # Unique users record with label and 'OPEN_MONTH'
users = users.rename(columns = {'MONTHS_BALANCE':'OPEN_MONTH'})
print(users['APPROVED'].value_counts())
users['APPROVED'].value_counts(normalize=True)

# Raw Data
# Merge two dataframes 'application' and 'open_month' by 'ID'
raw = pd.merge(application, users, how = 'right', on ='ID')
raw.dropna(inplace = True) # Drop credit records with no information - ANY
# data.dropna(subset = ['CODE_GENDER'], inplace = True) # Drop credit records with no information (not matching applications)

raw = raw.astype({'FLAG_WORK_PHONE': 'int64','FLAG_PHONE': 'int64','FLAG_EMAIL': 'int64'}) # Preserve binary features as integer
raw # Applications with matching credit records

# Raw Data records
raw.info()
raw.describe()
raw.nunique()
raw.iloc[:,:].agg(lambda x:';'.join(str(y) for y in x.unique())).to_frame('Unique Values')

# Copy raw data to 'data' dataframe ready to be cleaned
data = pd.DataFrame(raw['ID'])

"""# Features
---
"""

data

"""## Binary Features

### Gender (CODE_GENDER)
"""

map_array = {'F':0, 'M':1}
data['Gender'] = raw['CODE_GENDER'].astype(str).replace(map_array)
print(data['Gender'].value_counts())

"""### Car Ownership (FLAG_OWN_CAR)"""

map_array = {'N':0, 'Y':1}
data['Car'] = raw['FLAG_OWN_CAR'].astype(str).replace(map_array)
print(raw['FLAG_OWN_CAR'].value_counts())

"""### Realty Ownership (FLAG_OWN_REALTY)"""

map_array = {'N':0, 'Y':1}
data['Realty'] = raw['FLAG_OWN_REALTY'].astype(str).replace(map_array)
print(raw['FLAG_OWN_REALTY'].value_counts())

"""### Mobile Phone (FLAG_MOBIL)"""

# data.drop(labels = 'FLAG_MOBIL', axis = 1, inplace = True, errors = 'ignore') #Dropping column

"""### Work Phone (FLAG_WORK_PHONE)"""

map_array = {'N':0, 'Y':1}
data['Work Phone'] = raw['FLAG_WORK_PHONE'].astype(str).replace(map_array)
print(raw['FLAG_WORK_PHONE'].value_counts())

"""### Phone (FLAG_PHONE)"""

map_array = {'N':0, 'Y':1}
data['Phone'] = raw['FLAG_PHONE'].astype(str).replace(map_array)
print(raw['FLAG_PHONE'].value_counts())

"""### Email (FLAG_EMAIL)"""

map_array = {'N':0, 'Y':1}
data['Email'] = raw['FLAG_EMAIL'].astype(str).replace(map_array)
print(raw['FLAG_EMAIL'].value_counts())

"""---
## Continuous Features

### Number of Children (CNT_CHILDREN)
"""

data['# of Children'] = np.clip(raw['CNT_CHILDREN'], a_min = None, a_max = 2 ) # Clipping at 2 children and above
print(data['# of Children'].value_counts())
data['# of Children'].plot(kind='hist', xticks = [0,1,2])

"""### Annual Income (AMT_INCOME_TOTAL)

The currency is in CNY (1 USD = 0.16 CNY)
"""

raw['AMT_INCOME_TOTAL'].plot(kind='hist', bins = 50,density = True)