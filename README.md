# 💻 Laptop Price Predictor 🔮

A machine learning-powered web application that predicts the price of a laptop based on key specifications like brand, CPU, RAM, storage, GPU, screen resolution, and more.

## 🚀 Live Demo
Coming Soon...

## 📷 Screenshot
![Laptop Price Predictor UI](https://via.placeholder.com/800x400.png?text=Project+Screenshot)

---

## 🧠 Features

- 🔍 Predicts price using a trained ML model.
- 🧩 Takes in detailed laptop specs (RAM, brand, CPU, etc).
- 📱 Clean, responsive frontend using React + Vite.
- 🔁 Live prediction as inputs change.
- ✅ Error validation and real-time feedback.
- 🎯 Simple, fast, and easy-to-use interface.

---

## ⚙️ Tech Stack

**Frontend**
- HTML/CSS/JavaScript
- Toast notifications and loading spinner

**Backend**
- Python (Flask)
- Scikit-learn
- Pandas
- Pickled ML model (`pipe.pkl`)

---

## 🏗️ How It Works

1. User enters the laptop configuration.
2. The frontend calculates additional parameters like PPI.
3. Sends data to Flask backend via API.
4. Trained ML model processes the input.
5. Returns the predicted price to frontend.


---

## 🛠️ Setup & Installation

### 📦 Backend (Flask + ML Model)

cd backend
pip install -r requirements.txt
python app.py

cd frontend
run HTML file on live server using `Go Live`


🧪 Model Training
The model is trained on a dataset of laptop specifications and prices using a Pipeline with:

Label Encoding

Feature Transformation

Regression Algorithm (e.g., RandomForestRegressor)

##🧑‍💻 Author
###Onkar Shinde
AI & Data Science Enthusiast
## 📬 Let's Connect
[LinkedIn](https://www.linkedin.com/in/onkar-shinde77/)
