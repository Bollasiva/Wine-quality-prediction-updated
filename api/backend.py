from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__, template_folder="../templates")

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '../wine_mod.pkl')
model = pickle.load(open(model_path, 'rb'))

def interpret_quality(score):
    if score <= 3:
        return "Bad wine 👎 – Probably not your best pick!"
    elif 4 <= score <= 6:
        return "Average wine 😐 – Might go well with a casual dinner."
    elif 7 <= score <= 8:
        return "Good wine 👍 – Sounds like a tasty choice!"
    else:
        return "Excellent wine 🍷 – Pop that cork and enjoy!"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            input_features = [
                float(request.form['fixed_acidity']),
                float(request.form['volatile_acidity']),
                float(request.form['citric_acid']),
                float(request.form['residual_sugar']),
                float(request.form['chlorides']),
                float(request.form['free_sulfur_dioxide']),
                float(request.form['total_sulfur_dioxide']),
                float(request.form['density']),
                float(request.form['pH']),
                float(request.form['sulphates']),
                float(request.form['alcohol'])
            ]
            features = np.array([input_features])
            prediction = model.predict(features)
            predicted_quality = int(round(prediction[0]))
            message = interpret_quality(predicted_quality)

            return render_template('index.html',
                                   prediction_text=f'Predicted Wine Quality: {predicted_quality}/10',
                                   quality_message=message)

        except Exception as e:
            return render_template('index.html',
                                   prediction_text="Something went wrong!",
                                   quality_message=str(e))
    return render_template('index.html')

# Expose app for Vercel
from mangum import Mangum
handler = Mangum(app)
