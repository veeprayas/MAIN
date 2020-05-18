import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import cv2
import pickle
import numpy as np
from PIL import *
import tensorflow as tf
app = Flask(__name__)


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

@app.route('/camera',methods=['POST'])
def camera():
    f = request.files['file']
    f.save(f.filename)
    name = request.files['file'].filename

    # model = load_model('model.h5')
    model1 = pickle.load(open('model.pkl', 'rb'))

    # model1.summary()

    img = cv2.imread(name)
    img = cv2.resize(img, (299, 299))
    img = np.reshape(img, [1, 299, 299, 3])

    classe = model1.predict(img / 255)

    print(classe)
    return render_template('secondmain.html',prediction_text='The image was uploaded successfully')

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
    #output=model.predict([[a]])
    #print(a)
    #print(b)
    #print(c)
    df = pd.read_csv('npk.csv')
    # df = df.drop("SOP", axis=1)

    #print(df.shape)
    #print(df.head(10))
    #print(df.describe())
    # print(df.groupby('class').size())

    features = ['N', 'P', 'K']
    features1 = ['Urea', 'SOP']

    x = df.loc[:, features].values
    y = df.loc[:, features1].values

    # x = StandardScaler().fit_transform(x)
    # y = StandardScaler().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)
    #print(y_test)
    #print(X_test)

    plt.plot(X_train, y_train)
    plt.title('Graph')
    plt.xlabel('NPK2D', color='#1C2833')
    plt.ylabel('UREA', color='#1C2833')
    plt.legend(loc='upper left')
    plt.show()

    pls2 = PLSRegression(n_components=2)
    pls2.fit(X_train, y_train)
    PLSRegression()
    Y_pred = pls2.predict(X_test)

    #print(Y_pred)
    #print(np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))
    #print(np.sqrt(metrics.r2_score(y_test, Y_pred)))

    N = [a]
    P = [b]
    K = [c]
    list_of_tuples = list(zip(N, P, K))
    list_of_tuples
    df = pd.DataFrame(list_of_tuples)

    print(df)
    Y1_pred = pls2.predict(df)
    print(Y1_pred)

    pickle.dump(pls2, open('model8.pkl', 'wb'))

    # Loading model to compare the results
    model = pickle.load(open('model8.pkl', 'rb'))

    return render_template('firstmain.html', prediction_text=' Amount of urea and SOP to be added is {} respectively'.format(Y1_pred))

if __name__ == "__main__":
    app.run(debug=True)





    