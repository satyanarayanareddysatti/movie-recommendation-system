import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load datasets
movies = pd.read_csv("/content/movies.csv")
ratings = pd.read_csv("/content/ratings.csv")

# -----------------------------
# 🔹 Part 1: Content-Based Setup
# -----------------------------
movies['genres'] = movies['genres'].fillna('')
cv = CountVectorizer(tokenizer=lambda x: x.split('|'))
count_matrix = cv.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(count_matrix)

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_similar_movies(movie_title, num_recs=5):
    if movie_title not in indices:
        return ["Movie not found in dataset."]
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return [movies.iloc[i[0]]['title'] for i in sim_scores[1:num_recs+1]]

# ---------------------------------
# 🔹 Part 2: Collaborative Filtering (User-Item Matrix)
# ---------------------------------
# Build User-Item matrix
user_item_matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix.fillna(0))
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

movie_dict = dict(zip(movies['movieId'], movies['title']))

def get_user_recommendations(user_id, num_recs=5):
    if user_id not in user_item_matrix.index:
        return None

    # Get similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id).index

    # Get movies rated by similar users
    recommended_movies = []
    for sim_user in similar_users:
        sim_user_ratings = user_item_matrix.loc[sim_user].dropna()
        for movie_id in sim_user_ratings.index:
            if pd.isna(user_item_matrix.loc[user_id, movie_id]):  # unseen movie
                recommended_movies.append((movie_id, sim_user_ratings[movie_id]))

        if len(recommended_movies) >= num_recs:
            break

    # Return top movies
    recommended_movies = sorted(recommended_movies, key=lambda x: x[1], reverse=True)
    return [movie_dict[mid] for mid, _ in recommended_movies[:num_recs]]

# -------------------------------
# 🔹 Part 3: Popular / Genre Movies Fallback
# -------------------------------
def get_popular_movies(num_recs=5):
    popular = ratings.groupby("movieId").size().sort_values(ascending=False)[:num_recs].index
    return [movie_dict[mid] for mid in popular]

def get_genre_movies(preferred_genres, num_recs=5):
    filtered = movies[movies['genres'].apply(lambda g: any(genre in g for genre in preferred_genres))]
    return filtered.sample(min(num_recs, len(filtered)))['title'].tolist()

# -------------------------------
# 🔹 Part 4: Hybrid Recommender
# -------------------------------
def hybrid_recommend(user_id=None, movie_title=None, preferred_genres=None, num_recs=5):
    if movie_title:  # Case 1: Search by movie
        print(f"Because you searched '{movie_title}':")
        return get_similar_movies(movie_title, num_recs)

    elif user_id and user_id in user_item_matrix.index:  # Case 2: Known user
        print(f"Personalized recommendations for User {user_id}:")
        return get_user_recommendations(user_id, num_recs)

    elif preferred_genres:  # Case 3: New user selects genres
        print(f"Because you like {preferred_genres}:")
        return get_genre_movies(preferred_genres, num_recs)

    else:  # Case 4: Totally new user (no info)
        print("New user detected! Showing popular movies:")
        return get_popular_movies(num_recs)

