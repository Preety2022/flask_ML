from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('profit.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def start_app():
    return render_template('home.html')

@app.route('/search')
def predict():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['rnd']
    data2 = request.form['admin']
    data3 = request.form['marketing']
    arr = np.array([[data1, data2, data3]])
    user_input_prediction = arr.astype('int')
    pred = model.predict(user_input_prediction)
    pred = str(pred)
    pred = pred.replace('[','').replace(']','').strip()
    return render_template('predict.html', data=pred)


if __name__ == "__main__":
    app.debug = True
    app.run()