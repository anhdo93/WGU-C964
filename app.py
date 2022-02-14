from flask import Flask, render_template, request
import pandas as pd
import model_trained
from User import User

app = Flask(__name__)



@app.route("/")
def index():
    return render_template("index.html",score=500)

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


if __name__ == "__main__":
    app.run(debug=True)