import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle# -*- coding: utf-8 -*-

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = prediction
    def predict_text():
        if prediction ==0:
            print('Non-defaulter')
        else:
            print('Defaulter')
    
    return render_template('index.html',predict_text())
                         

if __name__ == '__main__':
    app.run(debug=True)


