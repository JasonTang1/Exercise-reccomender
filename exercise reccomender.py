import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import requests
import streamlit as st
from openai import OpenAI


# Set directory cd C:\Users\jason\OneDrive\Documents\CS303E
st.title('My Exercise Recommender')
st.write("This is a simple exercise recommender that uses RAG to recommend exercises based on your experience level and what you want to train today.") 

# Load in dataset
df = pd.read_csv('exercise_dataset_modified.csv')

# Make description column
def format_description(row):
    body_parts = [part.strip() for part in row['Body Part'].split(',')]
    if len(body_parts) > 1:
        body_part_str = ' and '.join(body_parts)
    else:
        body_part_str = body_parts[0]
    return f"{row['Exercise']} targets the {body_part_str} using {row['Equipment']} and is {row['Difficulty']} difficulty."

df['description'] = df.apply(format_description, axis=1)

# Turning descriptions into embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and fast
exercise_embeddings = model.encode(df['description'].tolist(), convert_to_tensor=True)

user_question = st.text_input("What is your experience level and what do you want to train today?")

if user_question.strip():  # Only proceed if input is non-empty/valid
    query_embedding = model.encode(user_question, convert_to_tensor=True)

    # Running cosine similarity
    similarities = cosine_similarity(query_embedding, exercise_embeddings)
    top_k_indices = similarities.topk(5).indices  # Top 5 results

    exercises_information = df.iloc[top_k_indices.tolist()]

    retrieved_exercises = "\n".join([
        f"- {row['Exercise']}: targets {row['Body Part']}, uses {row['Equipment']}, difficulty: {row['Difficulty']}"
        for _, row in exercises_information.iterrows()
    ])

    # print(retrieved_exercises)
    # Example user question: I'm a beginner. I want to do some easy leg exercises.

    rag_prompt = f"""
    The user's question is: "{user_question}".

    Based only on the following exercises:
    {retrieved_exercises}

    Please recommend exactly 4 distinct exercises that effectively target the muscle groups mentioned by the user. 

    Format your response as follows:
    1. List the four exercises in order. For each, include:
    - Exercise name
    - Targeted body part(s)
    - Equipment used
    - Difficulty level
    2. Then, create a detailed workout plan that includes:
    - A dynamic warm-up, including stretching and one lower-intensity warm-up set for each muscle group worked
    - The main workout sets with the recommended exercises, making sure total sets for each body part do not exceed 10
    - Clear rest times between sets: 3-4 minutes for large muscle groups (legs, chest), 2-3 minutes for smaller muscle groups (arms, shoulders)
    - A cooldown consisting of stretching only (no breathing exercises or other cooldowns needed)

    Important instructions:
    - Do not repeat the same exercise in the list or in the workout plan
    - Avoid recommending two similar exercises that require different equipment (e.g., do not mix barbell and dumbbell exercises for the same muscle group)
    - If the user is a beginner, prioritize easy-to-perform exercises
    - If the user is older, suggest low-impact exercises
    - If the user has no equipment, suggest bodyweight exercises
    - For warm-ups, only one warm-up exercise per muscle group should be included
    - Maintain an encouraging and helpful tone throughout
    """

    # Using llama3 for generating a response
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=st.secrets["GROQ_API_KEY"]
    )

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": rag_prompt}],
        temperature=0.7
    )

    st.write(response.choices[0].message.content)
else:
    st.write("Please enter your experience level and training goals above to get exercise recommendations.") #ask for user input if non given



# # Local llama3 server (uncomment to use local llama3 server)
# response = requests.post(
#     "http://localhost:11434/api/generate",
#     json={"model": "llama3", "prompt": rag_prompt, "stream":  False}
# )
#
# response_data = response.json()
# if 'response' in response_data:
#     print()
#     st.write(response_data['response']) 
# else:
#     st.write("Error or unexpected response:", response_data) 
