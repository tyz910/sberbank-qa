import csv
import random
import pickle
from util.model import QaPredictLiveModel
from flask import Flask, request, jsonify, render_template

APP_HOST = '0.0.0.0'
APP_PORT = 8000
app = Flask(__name__, template_folder="assets/templates", static_folder="assets/static")

with open('model/vocab.pickle', 'rb') as f:
    vocab = pickle.load(f)

with open('model/weights.pickle', 'rb') as f:
    weights = pickle.load(f)

model = QaPredictLiveModel({
    'hidden_size': 64,
    'doc_layers': 3,
    'question_layers': 3,
    'dropout_emb': 0.3,
    'dropout_rnn': 0.3,
    'cuda': False
}, vocab, weights)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_handler():
    data = request.get_json()
    predict = model.predict(data['paragraph'], data['question'])

    return jsonify({'answer': predict.answer.text})


@app.route('/random')
def random_handler():
    line = random.choice(open("data/train.csv").readlines())
    parts = list(csv.reader([line]))[0]
    return jsonify({
        "paragraph": parts[2],
        "question": parts[3],
    })


@app.route('/health')
def health_handler():
    return jsonify({'ok': True})


if __name__ == '__main__':
    app.run(host=APP_HOST, port=APP_PORT)
