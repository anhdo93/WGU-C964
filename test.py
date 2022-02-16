from flask import Flask, render_template, request
import pandas as pd

import model_trained
import cleaned_data_dashboard as dashboard
from User import User

import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import prince
import matplotlib.pyplot as plt

app = Flask(__name__)



df=dashboard.data()
all_columns = list(df.columns)
print(len(all_columns))
label = ['Rejected']
all_features = [feature for feature in all_columns if feature not in label]

features=all_features
# Prince method - from DOC
X=df[features]
#mca = prince.MCA(n_components=2, n_iter=3, copy=True, check_input=True, engine='auto',random_state=42)
#mca=mca.fit(X)
"""
ax = mca.plot_coordinates(
            X=X,ax=None, figsize=(6, 6), show_row_points=True, 
            row_points_size=10, show_row_labels=False, 
            show_column_points=True, column_points_size=30, show_column_labels=False, legend_n_cols=1)
"""
#fig = ax.get_figure().show()


# Prince show with matplotlib
#fig, ax = plt.subplots()
mca = prince.MCA(n_components=32).fit(X)
#mca.plot_coordinates(X=X, ax=ax, show_row_points=False)
#ax = mca.plot_coordinates(X=X,ax=None,figsize=(6, 6),show_row_points=True,row_points_size=10,show_row_labels=False, 
     #show_column_points=True,  column_points_size=30, show_column_labels=False, legend_n_cols=1)

#ax.set_xlabel('Component 1', fontsize=16)
#ax.set_ylabel('Component 2', fontsize=16)
#plt.show()


categories=mca.column_coordinates(df[features])[[0,1]]
fig = px.scatter(categories,title='MCA Features Plot',x=0,y=1,color=categories.index,labels={'0':'Dimension 0', '1':'Dimension 1'})
points=mca.transform(df[features])
points['Rejected'] = df[label]
#fig.add_traces(list(px.scatter(points,x=0,y=1)))
fig2 = px.scatter(points,x=0,y=1,color='Rejected',labels={'0':'Dimension 0', '1':'Dimension 1'})
print(points['Rejected'].unique())
fig.show()


"""
# Plotly PCA
fig = px.scatter_matrix(df, dimensions=features, color="Rejected")
fig.update_traces(diagonal_visible=False)
"""