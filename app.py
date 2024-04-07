from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("D:/MACHINE_LEARNING_PROJECTS/IRIS/irismodel.pkl")  # Use forward slashes or double backslashes

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        data = [float(request.form['feature1']),
                float(request.form['feature2']),
                float(request.form['feature3']),
                float(request.form['feature4'])]
        x_tst = np.array(data).reshape(1, -1)
        pred = model.predict(x_tst)
        pred_str = str(pred[0])  # Convert NumPy array to string
        return render_template('result.html', prediction=pred_str)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
