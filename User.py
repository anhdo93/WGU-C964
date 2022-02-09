import pandas as pd

class User:
    def __init__(self,features):
        for feature in features:
            setattr(self,feature,0)

    def to_dict(self,features):
        dict = {}
        for feature in features:
            dict[feature]=getattr(self,feature)
        return dict

    # Enable object support item assignment
    def __setitem__(self, key, value):
        setattr(self, key, value)

    # Enable object support item assignment
    def __getitem__(self, key):
        return getattr(self, key)
    
    def default(self):
        setattr(self,'Gender',1)                                        #0-Female 1-Male
        
        setattr(self,'Age_low',1)                                       #lowest(20-29)/low(30-39)/medium(40-49)/high(50-59)/highest(>60)
        setattr(self,'Education_'+'Secondary / secondary special',1)    #Secondary / secondary special/Higher education/Incomplete higher/Lower secondary 
        setattr(self,'Marital Status_'+'Married',1)                     #Married/Single / not married/Civil marriage/Separated/Widow 
        setattr(self,'# of Children_'+'0.0',1)                          #0.0/1.0/2.0
        setattr(self,'Family Size_'+'1.0',1)                            #1.0/2.0/3.0

        #setattr(self,'Employment Status']
        setattr(self,'Years Employed_'+'highest',1)                     #lowest/low/medium/high/highest
        setattr(self,'Annual Income_'+'high',1)                         #low/medium/high
        setattr(self,'Income Category_'+'Commercial associate',1)       #Industry - Commercial associate/Working/State servant
        setattr(self,'Occupation_'+'High Tech',1)                       #Labor/Office/High Tech
        setattr(self,'Residency_'+'House / apartment',1)                #House / apartment/With parents/Municipal apartment/Rented apartment/Office apartment/Co-op apartment  

        setattr(self,'Email',1)
        #setattr(self,'Mobile Phone'],1)
        setattr(self,'Work Phone',1)
        setattr(self,'Car',1)
        setattr(self,'Realty',1)


