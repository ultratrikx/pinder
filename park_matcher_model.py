import csv
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def load_csv(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Get the header row
        return header, list(reader)


def create_doc(text):
    return nlp(text.lower())


def create_user_vector(user):
    user_text = f"{user['hobbies']} {user['interests']} {user['favorite_activity']} {user['bio']}"
    return create_doc(user_text).vector


def create_park_vector(park, header):
    park_text = f"{park[header.index('Name')]} {park[header.index('Location')]} {park[header.index('Description')]}"
    return create_doc(park_text).vector


def create_model():
    # Load data
    parks_header, parks_data = load_csv('parks.csv')
    users_header, users_data = load_csv('users.csv')

    # Create vectors
    park_vectors = np.array([create_park_vector(park, parks_header) for park in parks_data])
    user_vectors = np.array([create_user_vector(dict(zip(users_header, user))) for user in users_data])

    return {
        'parks_header': parks_header,
        'parks': parks_data,
        'users_header': users_header,
        'users': [dict(zip(users_header, user)) for user in users_data],
        'park_vectors': park_vectors,
        'user_vectors': user_vectors
    }


def find_matches(model, user_index, top_n=5):
    user_vector = model['user_vectors'][user_index].reshape(1, -1)
    similarities = cosine_similarity(user_vector, model['park_vectors'])[0]
    top_indices = similarities.argsort()[-top_n:][::-1]

    matches = []
    for i in top_indices:
        park = model['parks'][i]
        matches.append({
            'name': park[model['parks_header'].index('Name')],
            'location': park[model['parks_header'].index('Location')],
            'description': park[model['parks_header'].index('Description')],
            'similarity': float(similarities[i])  # Convert to float for JSON serialization
        })

    return matches


if __name__ == "__main__":
    model = create_model()

    user_matches = find_matches(model, 0)

    print(f"Top 5 park matches for {model['users'][0]['name']}:")
    for match in user_matches:
        print(f"{match['name']} - Similarity: {match['similarity']:.2f}")
        print(f"Location: {match['location']}")
        print(f"Description: {match['description']}")
        print()

    # Save model (optional)
    np.save('park_vectors.npy', model['park_vectors'])
    np.save('user_vectors.npy', model['user_vectors'])
    print("Model vectors saved.")