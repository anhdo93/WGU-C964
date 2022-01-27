from flask import Flask, render_template, request
#import model

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    gender=request.form.get("GENDER")
    dob=request.form.get("DOB")
    education=request.form.get("EDUCATION")
    marital_status=request.form.get("MARITAL_STATUS")
    children_count=request.form.get("CHILDREN_COUNT")
    family_size=request.form.get("FAMILY_SIZE")
    
    employment_status=request.form.get("EMPLOYMENT_STATUS")
    employment_date=request.form.get("EMPLOYMENT_DATE")
    annual_income=request.form.get("ANNUAL_INCOME")
    industry=request.form.get("INDUSTRY")
    occupation=request.form.get("OCCUPATION")
    living_style=request.form.get("LIVING_STYLE")

    email=request.form.get("EMAIL")
    mobile_phone=request.form.get("MOBILE_PHONE")
    work_phone=request.form.get("WORK_PHONE")
    car=request.form.get("CAR")
    home=request.form.get("HOME")


    return gender
    


if __name__ == "__main__":
    app.run(debug=True)