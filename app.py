
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load("disaster_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    tweet = ""

    if request.method == "POST":
        tweet = request.form.get("tweet", "").strip()

        if tweet:
            # Convert tweet into numerical features
            tweet_vector = vectorizer.transform([tweet])

            # Predict tweet category
            result = int(model.predict(tweet_vector)[0])

            if result == 1:
                prediction = "Disaster-related"
            else:
                prediction = "Non-disaster"

    return render_template(
        "index.html",
        prediction=prediction,
        tweet=tweet
    )


if __name__ == "__main__":
    app.run(debug=True)
