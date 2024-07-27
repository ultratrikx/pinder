import csv
from flask import Flask, render_template, request, jsonify
import numpy as np
from park_matcher_model import create_model, find_matches, create_user_vector

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = create_model()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['POST'])
def register():
    user_data = request.json
    model['users'].append(user_data)
    user_vector = create_user_vector(user_data)
    model['user_vectors'] = np.vstack([model['user_vectors'], user_vector])

    with open('users.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_data[field] for field in model['users_header']])

    return jsonify({"status": "success"})


@app.route('/match/<username>')
def match(username):
    user_index = next((i for i, u in enumerate(model['users']) if u['name'] == username), None)
    if user_index is None:
        return jsonify({"error": "User not found"})

    matches = find_matches(model, user_index)
    return jsonify(matches)


@app.route('/swipe', methods=['POST'])
def swipe():
    swipe_data = request.json
    return jsonify({"status": "success"})


if __name__ == '__main__':
    app.run(debug=True)