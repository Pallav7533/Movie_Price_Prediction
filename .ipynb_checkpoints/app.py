from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('movie_rating_prediction_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    # Assuming the form contains fields with names 'year', 'duration', and 'votes'
    new_input = [[int(data['year']), int(data['duration']), int(data['votes'])]]
    predicted_rating = model.predict(new_input)
    return render_template('index.html', prediction=predicted_rating[0])

if __name__ == '__main__':
    app.run(debug=True)
