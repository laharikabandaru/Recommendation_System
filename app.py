import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template

# Sample Data Preparation
data = {
    'user_id': [1, 1, 2, 2, 3, 3],
    'item_id': [101, 102, 101, 103, 104, 105],
    'rating': [5, 4, 5, 4, 4, 5]
}
interactions = pd.DataFrame(data)

# Additional Item Data (for example, item names or descriptions)
items_data = {
    'item_id': [101, 102, 103, 104, 105],
    'item_name': ['Item A', 'Item B', 'Item C', 'Item D', 'Item E']
}
items = pd.DataFrame(items_data)

# Create User-Item Matrix
user_item_matrix = interactions.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Compute the similarity matrix
similarity = cosine_similarity(user_item_matrix)

# Recommendation Function
def recommend(user_id, user_item_matrix, similarity, top_n=5):
    try:
        user_index = user_id - 1  # Adjust for 0-based index
        scores = similarity[user_index] @ user_item_matrix.values
        scores /= np.sum(similarity[user_index])  # Normalize the scores
        recommended_items = np.argsort(scores)[::-1][:top_n]
        recommended_item_ids = user_item_matrix.columns[recommended_items].tolist()
        return recommended_item_ids
    except IndexError:
        return []

# Function to get item names from item IDs
def get_item_names(item_ids):
    item_names = items[items['item_id'].isin(item_ids)]['item_name'].tolist()
    return item_names

# Flask App Setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend_endpoint():
    try:
        user_id = int(request.args.get('user_id'))
        
        if user_id <= 0:
            return jsonify({'error': 'Invalid user_id, must be a positive integer.'}), 400
        
        recommendations = recommend(user_id, user_item_matrix, similarity)
        
        if not recommendations:
            return jsonify({'error': 'No recommendations available for this user.'}), 404
        
        # Get item names for the recommended item IDs
        recommended_item_names = get_item_names(recommendations)
        
        return render_template('recommendations.html', user_id=user_id, recommended_items=recommended_item_names)
    
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid input, user_id must be an integer.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
