import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
#import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import statsmodels.formula.api as smf

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))
model = pickle.load(open('model8.pkl','rb'))
@app.route('/')
def home():
    return render_template('registration.html')

@app.route('/mainpage',methods=['GET', 'POST'])
def mainpage():
    #return render_template('mainpage.html')
    user = request.form.get('u')
    pas = request.form.get('p')
    if user=='ninad' and pas=='1234':
        return render_template('mainpage.html')
    else:
        return render_template('registration.html')
    #if request.method == 'POST':
     #   if request.form['password'] == '1234' and request.form['username'] == 'ninad':
      #      return render_template('mainpage.html')
       # else:
        #    return render_template('registration.html')

@app.route('/firstmain',methods=['GET'])
def firstmain():
    return render_template('firstmain.html')

@app.route('/secondmain',methods=['GET'])
def secondmain():
   return render_template('secondmain.html')

#@app.route('/camera',methods=['POST'])
#def camera():

 #   f = request.files['file']
  #  f.save(f.filename)
   # name = request.files['file'].filename
    #print(name)
    #return render_template('secondmain.html',prediction_text='The image was uploaded successfully')

@app.route('/prediction',methods=['POST'])
def prediction():
    '''
    For rendering results on HTML GUI
    '''
    select=request.form.get('n')
    a=int(select)
    select1 = request.form.get('p')
    b = int(select1)
    select2 = request.form.get('k')
    c = int(select2)



    N = [a]
    P = [b]
    K = [c]
    list_of_tuples = list(zip(N, P, K))
    list_of_tuples
    df = pd.DataFrame(list_of_tuples)

    print(df)
    Y1_pred = model.predict(df)
    print(Y1_pred)



    return render_template('firstmain.html', prediction_text=' Amount of urea and SOP to be added is {} respectively'.format(Y1_pred))

if __name__ == "__main__":
    app.run(debug=True)