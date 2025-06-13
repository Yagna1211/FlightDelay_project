from flask import Flask, render_template, request
import numpy as np 
import pickle 

# Load the trained model
model = pickle.load(open("flight.pkl", 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def predict():
    # Collect numeric inputs
    flnum = int(request.form['flnum'])
    month = int(request.form['month'])
    dayofmonth = int(request.form['dayofmonth'])
    dayofweek = int(request.form['dayofweek'])
    crsarr = int(request.form['crsarr'])
    dep15 = int(request.form['dep15'])

    # One-hot encode origin
    origin = request.form['origin']
    origin1, origin2, origin3, origin4, origin5 = 0, 0, 0, 0, 0
    if origin == "dtw": origin1 = 1
    elif origin == "sea": origin2 = 1
    elif origin == "jfk": origin3 = 1
    elif origin == "atl": origin4 = 1
    elif origin == "msp": origin5 = 1

    # One-hot encode destination
    destination = request.form['destination']
    destination1, destination2, destination3, destination4, destination5 = 0, 0, 0, 0, 0
    if destination == "dtw": destination1 = 1
    elif destination == "sea": destination2 = 1
    elif destination == "jfk": destination3 = 1
    elif destination == "atl": destination4 = 1
    elif destination == "msp": destination5 = 1

    # Form the complete feature array (16 total features)
    total = [[
        flnum, month, dayofmonth, dayofweek, crsarr, dep15,  
        origin1, origin2, origin3, origin4, origin5,        
        destination1, destination2, destination3, destination4, destination5  
    ]]

    # Predict using the model
    y_pred = model.predict(total)

    # Interpret the result
    ans = "The Flight Will be on time" if y_pred[0] == 0 else "The flight will be delayed"
    
    return render_template("index.html", showcase=ans)

if __name__ == '__main__':
    app.run(debug=True)
