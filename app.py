from flask import Flask, render_template, request
#import model

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    
    user['gender']=request.form.get("GENDER")
    user['dob']=request.form.get("DOB")
    user['education']=request.form.get("EDUCATION")
    user['marital_status']=request.form.get("MARITAL_STATUS")
    user['children_count']=request.form.get("CHILDREN_COUNT")
    user['family_size']=request.form.get("FAMILY_SIZE")
    
    user['employment_status']=request.form.get("EMPLOYMENT_STATUS")
    user['employment_date']=request.form.get("EMPLOYMENT_DATE")
    user['annual_income']=request.form.get("ANNUAL_INCOME")
    user['industry']=request.form.get("INDUSTRY")
    user['occupation']=request.form.get("OCCUPATION")
    user['living_style']=request.form.get("LIVING_STYLE")

    user['email']=request.form.get("EMAIL")
    user['mobile_phone']=request.form.get("MOBILE_PHONE")
    user['work_phone']=request.form.get("WORK_PHONE")
    user['car']=request.form.get("CAR")
    user['home']=request.form.get("HOME")


    return gender
    


if __name__ == "__main__":
    app.run(debug=True)