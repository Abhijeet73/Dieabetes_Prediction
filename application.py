from flask import Flask,request,render_template,app,Response,Request
import pandas as pd
import numpy as np
import pickle
application=Flask(__name__)
app=application

model=pickle.load(open("E:\All project file\Dieabetes_Prediction\model\modelForPrediction.pkl","rb"))
scaler=pickle.load(open("E:\All project file\Dieabetes_Prediction\model\standardScalar.pkl","rb"))

##route for home page 
@app.route('/')
def index():
    return render_template('index.html')

## route for single datapoint
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':
        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        Glicose = float(request.form.get('Glicose'))
        
        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Glicose]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'
            
        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")