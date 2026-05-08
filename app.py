from flask import Flask, render_template, request
import pickle

# Load model and vectorizer
model = pickle.load(open("news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    data = vectorizer.transform([news])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "REAL NEWS"
    else:
        result = "FAKE NEWS"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
    
