from flask import Flask, render_template, request, flash
from model import load_model, predict_rainfall, prepare_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    city_name = request.form['city_name']
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])

    try:
        # Prepare the model (train if not exists)
        prepare_model(city_name)

        # Load the trained model
        model = load_model(city_name)

        # Use a default value for last rainfall (e.g., 0 mm)
        last_rainfall = 0.0  

        # Get predicted rainfall and chance of rain
        predicted_rainfall, chance_of_rain = predict_rainfall(model, year, month, day, last_rainfall)

        # Passing all details to the results template
        return render_template('results.html', 
                               city_name=city_name,
                               year=year,
                               month=month,
                               day=day,
                               message=f'{predicted_rainfall:.2f} mm', 
                               chance_of_rain=f'{chance_of_rain:.2f}%')
    
    except ValueError as e:
        flash(str(e))
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
