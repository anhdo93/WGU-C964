# For machine learning
from xml.sax.handler import all_features
import lightgbm as lgb
import pandas as pd
from User import User


bst = lgb.Booster(model_file='preloaded_model.txt')


all_features=['Gender', 'Car', 'Realty', 'Work Phone', 'Phone', 'Email', 
'# of Children_0.0', '# of Children_1.0', '# of Children_2.0', 
'Annual Income_low', 'Annual Income_medium', 'Annual Income_high', 
'Age_lowest', 'Age_low', 'Age_medium', 'Age_high', 'Age_highest', 
'Years Employed_lowest', 'Years Employed_low', 'Years Employed_medium', 'Years Employed_high', 'Years Employed_highest', 
'Family Size_1.0', 'Family Size_2.0', 'Family Size_3.0', 
'Income Category_Commercial associate', 'Income Category_State servant', 'Income Category_Working', 
'Occupation_High Tech', 'Occupation_Labor', 'Occupation_Office', 
'Education_Higher education', 'Education_Incomplete higher', 'Education_Lower secondary', 'Education_Secondary / secondary special', 
'Marital Status_Civil marriage', 'Marital Status_Married', 'Marital Status_Separated', 'Marital Status_Single / not married', 'Marital Status_Widow', 
'Residency_Co-op apartment', 'Residency_House / apartment', 'Residency_Municipal apartment', 'Residency_Office apartment', 'Residency_Rented apartment', 'Residency_With parents']

user = User(all_features)
user.default()
user_df = pd.DataFrame(user.to_dict(all_features), index=[0])

def show_result(user_df):
  global bst
  return bst.predict(user_df)[0]

def get_features():
  return all_features
  
print(show_result(user_df))