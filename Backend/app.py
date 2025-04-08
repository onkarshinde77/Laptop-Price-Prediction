from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

pipe = pickle.load(open("pipe.pkl", "rb"))
data = pickle.load(open("data.pkl", "rb"))

@app.route("/options", methods=["GET"])
def get_options():
    return jsonify({
        "Company": sorted(data["Company"].unique().tolist()),
        "TypeName": sorted(data["TypeName"].unique().tolist()),
        "Ram": sorted(data["Ram"].unique().tolist()),
        "Cpu brand": sorted(data["Cpu brand"].unique().tolist()),
        "HDD" : sorted(data["HDD"].unique().tolist()),
        "SSD" : sorted(data["SSD"].unique().tolist()),
        "Gpu": sorted(data["Gpu"].unique().tolist()),
        "OpSys": sorted(data["OpSys"].unique().tolist()),
    })

@app.route("/predict", methods=["POST"])
def predict():
    content = request.json
    df = pd.DataFrame([content])
    prediction = np.exp(pipe.predict(df)[0])
    return jsonify({"price": round(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
