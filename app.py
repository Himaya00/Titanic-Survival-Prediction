from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Load the trained model
model_file = os.path.join(script_dir, 'classifier.pkl')
try:
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("Error: The classifier file ('model.pkl') is not found.")


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/Titanic", methods=['POST','GET'])
def Survival():
    try:
        Pclass=int(request.form['Pclass'])
        Sex=int(request.form['Sex'])
        Age=int(request.form['Age'])
        Fare=int(request.form['Fare'])
        Embarked=float(request.form['Embarked'])

        ss = model.predict([[Pclass, Sex, Age, Fare, Embarked]])
    except ValueError:
        return render_template('index.html', error="Invalid input! Please enter correct data.")

    return render_template('index.html', Survived=ss)

if __name__ == '__main__':
    app.run(debug=True)
