from flask import Flask, render_template, request
import pandas as pd

import model_trained
import cleaned_data_dashboard as dashboard
from User import User

import json
import plotly
import plotly.express as px
import plotly.graph_objects as go


app = Flask(__name__)



df=dashboard.data()
all_columns = list(df.columns)
label = ['Rejected']
all_features = [feature for feature in all_columns if feature not in label]


@app.route("/")
def index():
    return render_template("index.html", title="Home")

@app.route("/result", methods=["POST"])
def result():
    all_features = model_trained.get_features()    
    user = User(all_features)

    def assign(user,feature,html_label):
        if feature=='Gender':
            user[feature]=int(request.form.get(html_label))
            return
        elif feature in ['Email','Phone','Work Phone','Car','Realty']:
            user[feature]=1 if request.form.get(html_label)=="on" else 0
            return                       
        user[feature+'_'+request.form.get(html_label)]=1

    assign(user,'Gender','GENDER')
    assign(user,'Age','AGE')
    assign(user,'Education','EDUCATION')
    assign(user,'Marital Status','MARITAL_STATUS')
    assign(user,'# of Children','CHILDREN_COUNT')
    assign(user,'Family Size','FAMILY_SIZE')

    assign(user,'Years Employed','YEARS_EMPLOYED')
    assign(user,'Annual Income','ANNUAL_INCOME')
    assign(user,'Income Category','INCOME_CATEGORY')
    assign(user,'Occupation','OCCUPATION')
    assign(user,'Residency','RESIDENCY')

    assign(user,'Email','EMAIL')
    assign(user,'Phone','PHONE')
    assign(user,'Work Phone','WORK_PHONE')
    assign(user,'Car','CAR')
    assign(user,'Realty','HOME')

    user_df = pd.DataFrame(user.to_dict(all_features), index=[0])

    min=int(request.form.get("MINIMUM"))
    max=int(request.form.get("MAXIMUM"))
    approval=int(request.form.get("APPROVAL"))

    prediction = float(model_trained.show_result(user_df))
    score = round((1-prediction)*(max-min)+min)
    return str(score)
    #return user.to_dict(all_features)  
     
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", title="Dashboard", graphJSON1=plotPie(df), graphJSON2=plotBar(df), feature1=all_features, feature2=all_features)

@app.route('/cbPie', methods=['GET'])
def cbPie():
    return plotPie(df, request.args.get('data'))

@app.route('/cbBar', methods=['GET'])
def cbBar():
    return plotBar(df, request.args.get('data'))

@app.route('/callback', methods=['GET'])
def callback():
    return plotPie(df, request.args.get('data'))

def plotPie(df, feature="Gender"):
    fig = px.pie(df, names=feature)
    fig.update_traces(hovertemplate = "<b>%{label}</b> <br> %{value:,.r} <br> %{percent:.2%}%")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def plotBar(df, feature="Gender"):
    fig = go.Figure()
    for category in df[feature].unique():
        feature_approved = df[(df['Rejected']==0) & (df[feature]==category)].count()[feature]
        feature_rejected = df[(df['Rejected']==1) & (df[feature]==category)].count()[feature]
        fig.add_trace(go.Bar(x=['Approved', 'Rejected'],y=[feature_approved, feature_rejected], customdata=[feature_approved, feature_rejected], 
                                name=category, hovertemplate = "<b>%{label}</b> <br> %{customdata:,.r} <br> %{y:.2f}%"))
    fig.update_layout(barmode='stack', barnorm='percent', yaxis_title="Percent")  
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
    

if __name__ == "__main__":
    app.run(debug=True)