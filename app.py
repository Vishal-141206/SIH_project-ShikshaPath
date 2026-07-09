from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import logging
import os
import gdown
import json
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rapidfuzz import fuzz, process   # ✅ added for fuzzy matching

logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder="templates", static_folder="static")

# ========== GOOGLE DRIVE MODEL DOWNLOADER ==========
MODEL_DIR = "flan_t5_base"
FILES = {
    "pytorch_model.bin": "1ztNY9ELyLeT02ui5hyAEv1A9VqDcnMr6",
    "config.json": "12xVmdmMu0seNrJOe1WKo4T8_UQVZwhJk",
    "tokenizer.json": "1sGc2ah0XRi88YDraPhnP32z-1O9dwO30",
    "tokenizer_config.json": "1g8ufnCYM92ZKVFv0a34yFq7ZVAt0xyNo"
}

def download_model():
    """Download model files from Google Drive if not already present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename, file_id in FILES.items():
        out_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(out_path):
            print(f"⬇️ Downloading {filename} ...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, out_path, quiet=False)
    print("✅ Model is ready at:", MODEL_DIR)


# ========== LOCAL LLM (Chatbot) ==========
class LocalLLM:
    def __init__(self):
        download_model()  # ensure model is available
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

    def get_response(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Initialize chatbot
llm = LocalLLM()

# ========== LOAD CAREER DATA JSON ==========
CAREER_DATA_FILE = "career_data.json"
career_data = []

if os.path.exists(CAREER_DATA_FILE):
    with open(CAREER_DATA_FILE, "r", encoding="utf-8") as f:
        career_data = json.load(f)
        logging.info("Loaded career_data.json with %d entries", len(career_data))
else:
    logging.warning("career_data.json not found!")

# ========== QUIZ ML MODEL LOADER ==========
MODEL_PATH = "svm_model.pkl"
LABEL_PATH = "label_encoder.pkl"

model = None
label_encoder = None

def load_models():
    """Load ML models only once (lazy loading)."""
    global model, label_encoder
    if model is None or label_encoder is None:
        try:
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
                logging.info("Loaded model from %s", MODEL_PATH)
            else:
                logging.warning("Model file not found at %s", MODEL_PATH)

            if os.path.exists(LABEL_PATH):
                label_encoder = joblib.load(LABEL_PATH)
                logging.info("Loaded label encoder from %s", LABEL_PATH)
            else:
                logging.warning("Label encoder file not found at %s", LABEL_PATH)
        except Exception as e:
            logging.exception("Failed to load model or label encoder: %s", e)


# ========== QUIZ QUESTIONS ==========
questions = [
    "🔧 Do you enjoy hands-on activities like fixing gadgets, repairing things, or working with mechanical tools?",
    "🌳 Do you prefer spending time outdoors and being physically active rather than sitting indoors for long hours?",
    "🧩 Do you like solving complex problems, puzzles, or understanding how things work at a deeper level?",
    "🔬 Are you interested in conducting experiments, doing research, or exploring abstract scientific ideas?",
    "🎨 Do you feel fulfilled when you express yourself creatively through art, music, writing, or design?",
    "🕒 Do you enjoy having flexibility in your work schedule rather than following a strict routine?",
    "❤️ Do you feel happy when helping, teaching, or taking care of others?",
    "👂 Are you good at listening to people and helping them resolve conflicts or problems?",
    "👔 Do you enjoy leading teams, persuading others, or taking on business challenges?",
    "🚀 Do you feel ambitious and motivated to take risks in order to achieve bigger goals?",
    "📋 Do you like working in an organized environment with clear rules, predictable tasks, and step-by-step processes?",
    "📊 Do you enjoy managing data, handling budgets, or keeping detailed records and reports?"
]

dimension_map = [
    "Realistic", "Realistic", "Investigative", "Investigative",
    "Artistic", "Artistic", "Social", "Social",
    "Enterprising", "Enterprising", "Conventional", "Conventional"
]

degree_map = {
    "Science": [("B.Tech in CS", "💻"), ("B.Sc Physics", "⚛️"), ("MBBS", "🩺"), ("BCA", "🖥")],
    "Commerce": [("B.Com Hons", "📚"), ("CA", "🧾"), ("BBA Finance", "💹"), ("BA Economics", "💵")],
    "Arts": [("BA Psychology", "🧠"), ("BFA", "🎨"), ("BA Journalism", "📰"), ("BA English Lit", "📖")],
    "Vocational": [("Diploma Web Designing", "💻"), ("ITI (Electrical)", "⚡"), ("B.Voc Hospitality", "🏨"), ("Skill Plumbing", "🔧")]
}

career_map = {
    "Science": [("Software Engineer","💻"),("Research Scientist","🔬"),("Doctor","🩺"),("Data Scientist","📊")],
    "Commerce": [("Accountant","📒"),("Investment Banker","💰"),("Entrepreneur","🚀"),("Financial Analyst","📈")],
    "Arts": [("Psychologist","🧠"),("Graphic Designer","🎨"),("Journalist","📰"),("Content Writer","✍️")],
    "Vocational": [("Full-Stack Dev","💻"),("Electrician","⚡"),("Hotel Manager","🏨"),("Mechanic","🔧")]
}

# ========== ROUTES ==========
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/mapping")
def path_mapping():
    return render_template("mapping.html")

@app.route("/bot")
def bot_page():
    return render_template("bot.html")

@app.route("/college_map")
def college_map():
    return render_template("college_map.html")

@app.route("/mentor")
def mentor():
    return render_template("mentor.html")

@app.route("/exam_tracker")
def tracker():
    return render_template("exam_tracker.html")

@app.route("/resource")
def resource():
    return render_template("resource.html")



@app.route("/quiz")
def quiz():
    return render_template("quiz.html", questions_json=questions)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_models()
        if model is None or label_encoder is None:
            return jsonify({'error': 'Model or label encoder not loaded on server. Check server logs.'}), 500

        body = request.get_json(force=True)
        answers = body.get('answers')

        if not isinstance(answers, list):
            return jsonify({'error': 'answers must be a list'}), 400
        if len(answers) != len(questions):
            return jsonify({'error': f'Provide {len(questions)} answer values (1-5). Received {len(answers)}.'}), 400

        answers_int = [int(x) for x in answers]
        X = np.array([answers_int])
        pred_encoded = model.predict(X)
        pred_text = label_encoder.inverse_transform(pred_encoded)[0]

        dimension_scores = {}
        for i, dim in enumerate(dimension_map):
            dimension_scores[dim] = dimension_scores.get(dim, 0) + int(answers_int[i])

        messages = {
            "Science":"🔬 Explore, experiment, and innovate!",
            "Commerce":"💰 Learn financial thinking and business basics.",
            "Arts":"🎨 Grow your creative and communication skills.",
            "Vocational":"🔧 Develop hands-on and employable skills."
        }

        careers = career_map.get(pred_text, [])
        degrees = degree_map.get(pred_text, [])

        response = {
            'recommendation': pred_text,
            'message': messages.get(pred_text, ''),
            'careers': careers,
            'degrees': degrees,
            'dimension_scores': dimension_scores
        }
        return jsonify(response)

    except Exception as e:
        logging.exception("Error in /predict")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


# ========== FIXED GET_ANSWER ROUTE ==========
def clean_text(text):
    return re.sub(r"[^\w\s]", "", text.lower().strip())

@app.route("/get_answer", methods=["POST"])
def get_answer():
    user_question = request.json.get("question", "")
    if not user_question:
        return jsonify({"answer": "Please ask a valid question."}), 400

    if not career_data:
        return jsonify({"answer": "Career data is not available."}), 500

    # Prepare list of questions
    questions = [item["question"] for item in career_data]

    # Fuzzy match
    best_match, score, idx = process.extractOne(
        clean_text(user_question),
        [clean_text(q) for q in questions],
        scorer=fuzz.token_sort_ratio
    )

    if score > 70:  # similarity threshold
        answer = career_data[idx]["answer"]
    else:
        answer = "Sorry, I don’t have an exact answer for that."

    return jsonify({"answer": answer})


# ========== MAIN ==========
if __name__ == "__main__":
    app.run(debug=True)






























