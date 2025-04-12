import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load data and model
df = pd.read_csv("university.csv", encoding='latin-1')  # Adjust the path as needed
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Generate embeddings
descriptions = df["description"].tolist()
embeddings = model.encode(descriptions, convert_to_tensor=True)
df["embedding"] = [embedding.tolist() for embedding in embeddings]

# Function for recommending universities
def recommend_universities(student_profile, subject=None, top_n=5):
    student_embedding = model.encode(student_profile, convert_to_tensor=True)
    cosine_scores = util.cos_sim(student_embedding, embeddings)[0]
    df["score"] = cosine_scores.cpu().numpy()

    if subject:
        filtered_df = df[df["Top Fields of Study"].str.contains(subject, case=False, na=False)]
        top_matches = filtered_df.sort_values("score", ascending=False).head(top_n)
    else:
        top_matches = df.sort_values("score", ascending=False).head(top_n)

    return top_matches[["University Name", "Country", "score", "Top Fields of Study"]]

# Streamlit UI
st.title('University Recommendation System')
student_profile_input = st.text_area("Enter your profile or field of interest:")

subject_input = st.text_input("Enter subject of interest (e.g., robotics, AI, finance, etc.):")

if st.button('Get Recommendations'):
    if student_profile_input:
        st.write("Top 5 Global Universities:")
        global_top_5 = recommend_universities(student_profile_input, top_n=5)
        st.write(global_top_5)

        if subject_input:
            st.write(f"\nTop 5 {subject_input.capitalize()} Universities:")
            subject_specific_top_5 = recommend_universities(student_profile_input, subject=subject_input, top_n=5)
            st.write(subject_specific_top_5)
    else:
        st.write("Please enter a student profile!")
