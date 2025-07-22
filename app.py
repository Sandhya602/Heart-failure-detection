from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('heart_failure_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    input_data = np.array([features])
    prediction = model.predict(input_data)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

