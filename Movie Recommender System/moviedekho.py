import pickle
import pandas as pd
import streamlit as st

def recommend(movie):
    if movie not in movies['title'].values:
        return ["Movie not found! Please try another movie."]

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movie_names = []
    
    for i in movies_list:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        # recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)
        

    return recommended_movie_names

movies_dict= pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl','rb'))


st.title('Movie Recommander')
selected_movie_names= st.selectbox(
    'Enter movie name ',
    movies['title'].values 
)

if st.button('Recommend'):
    recommendation = recommend(selected_movie_names)
    for i in recommendation:
        st.write(i)
