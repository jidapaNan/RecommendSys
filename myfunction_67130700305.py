# Function to get movie recommendations for a user
def get_movie_recommendations(user_id, user_similarity_df, user_movie_ratings, n_recommendations):
    similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False).index[1:]
    user_ratings = user_movie_ratings.loc[user_id]
    
    recommendations = []
    for other_user in similar_users:
        other_user_ratings = user_movie_ratings.loc[other_user]

        movies_rating = other_user_ratings[(other_user_ratings >3) & (user_ratings==0)]
        movies = movies_rating.sort_values(ascending=False).index
        recommendations.extend(movies)
        if len(recommendations)>n_recommendations:
            break;

    return recommendations[:n_recommendations]  # Return top n recommendations

