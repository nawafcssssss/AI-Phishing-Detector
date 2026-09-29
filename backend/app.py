from flask import Flask, render_template, request, jsonify
import joblib
import re
from urllib.parse import urlparse
import os
app = Flask(__name__, template_folder="../frontend")

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "..", "model", "phishing_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "..", "model", "vectorizer.pkl"))
#model = joblib.load("model/phishing_model.pkl")
#vectorizer = joblib.load("model/vectorizer.pkl")


def calculate_url_risk(url):
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    score = 0

    # Suspicious URL features
    if re.search(r"https?://\d+\.\d+\.\d+\.\d+", url):
        score += 30

    if "@" in url:
        score += 20

    if len(url) > 100:
        score += 15
    elif len(url) > 75:
        score += 10

    if "xn--" in url:
        score += 15

    suspicious_words = [
        "login",
        "verify",
        "password",
        "account",
        "secure",
        "update",
        "confirm"
    ]

    for word in suspicious_words:
        if word in url.lower():
            score += 5

    # Too many subdomains
    if len(host.split(".")) >= 4:
        score += 10

    return min(score, 100)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    url = data.get("url", "").strip()

    if not url:
        return jsonify({
            "result": "يرجى إدخال رابط",
            "risk_score": 0
        })

    # AI prediction
    url_vector = vectorizer.transform([url])

    prediction = model.predict(url_vector)[0]
    phishing_probability = model.predict_proba(url_vector)[0][1] * 100

    # URL feature risk
    feature_risk = calculate_url_risk(url)

    # Combine AI + URL features
    risk_score = round(
        (phishing_probability * 0.75) +
        (feature_risk * 0.25)
    )

    if risk_score >= 70:
        result = "الرابط عالي الخطورة وقد يكون تصيدًا احتياليًا"
    elif risk_score >= 40:
        result = "الرابط يحتوي على مؤشرات تستحق الحذر"
    else:
        result = "لم يتم اكتشاف مؤشرات تصيد واضحة"

    return jsonify({
        "url": url,
        "result": result,
        "risk_score": risk_score
    })


if __name__ == "__main__":
    app.run(debug=True)