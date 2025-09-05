# 🎬 Movie Recommendation System (Hybrid)

This is a **Hybrid Movie Recommendation System** that combines:
- ✅ Content-Based Filtering (using movie genres)
- ✅ User-Based Collaborative Filtering (cosine similarity on user ratings)
- ✅ Cold-start handling for new users (popular movies / genre-based suggestions)

## 🚀 Features
- Search by movie → recommends similar movies
- Personalized recommendations for known users
- Genre-based suggestions for new users
- Popular movie fallback for totally new users

## 📂 Dataset
- MovieLens dataset (movies.csv, ratings.csv)

## ⚙️ Tech Stack
- Python
- Pandas, Scikit-learn

## ▶️ Run the Project
```bash
pip install -r requirements.txt
python recommender.py
