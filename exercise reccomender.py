import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
import requests
import streamlit as st
from openai import OpenAI


# Set directory cd C:\Users\jason\OneDrive\Documents\CS303E
st.title('My Exercise Recommender')
st.write("This is a simple exercise recommender that uses RAG to recommend exercises based on your experience level and what you want to train today.") 

#Load in dataset
df = pd.read_csv('exercise_dataset_modified.csv')

#Make description column
def format_description(row):
    body_parts = [part.strip() for part in row['Body Part'].split(',')]
    if len(body_parts) > 1:
        body_part_str = ' and '.join(body_parts)
    else:
        body_part_str = body_parts[0]
    return f"{row['Exercise']} targets the {body_part_str} using {row['Equipment']} and is {row['Difficulty']} difficulty."

df['description'] = df.apply(format_description, axis=1)

#Turning descriptions into embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and fast
exercise_embeddings = model.encode(df['description'].tolist(), convert_to_tensor=True)

user_question = st.text_input("What is your experience level and what do you want to train today?")
query_embedding = model.encode(user_question, convert_to_tensor=True)

#Running cosine similarity
similarities = cosine_similarity(query_embedding, exercise_embeddings)
top_k_indices = similarities.topk(5).indices  # Top 5 results

exercises_information = df.iloc[top_k_indices.tolist()]

retrieved_exercises = "\n".join([
    f"- {row['Exercise']}: targets {row['Body Part']}, uses {row['Equipment']}, difficulty: {row['Difficulty']}"
    for _, row in exercises_information.iterrows()
])

#print(retrieved_exercises)
#Example user question: I'm a beginner. I want to do some easy leg exercises.

rag_prompt = f"""
The user's question is "{user_question}".
Given only these exercises:
{retrieved_exercises}, reccomend the user 4 possible exercises to train that muscle group and also form a possible workout plan involving those exercises.

Format: List the four exercises in order, displaying the exercise name, the body part it targets, the equipment used, and the difficulty level. Then output the workout plan first with the warmups, and then the actual sets of exercises. 
Don't repeat the same exercise for multiple times or lines.

For cooldown: Stretching is enough, no need to take a few deep breaths or anything like that.

Tips to use when generating responses: 
- If the user is a beginner, suggest exercises that are easy to perform.
- If the user is older, suggest low-impact exercises.
- If the user has no equipment, suggest bodyweight exercises.
- For workout plans, suggest a dynamic warm-up involving some stretching. Also reccomend one additional set of the same exercise, but at a lower intensity at the start to warmup the muscles in use. Only reccomend one warmup exercise for each body part.
- For a single body part workout, the total sets for that body part should not exceed 10 sets.
- For workout plan, don't reccomend two similar exercises that require different equipment, such as a barbell and a dumbbell, for the same body part. 
- For rests, suggest 3-4 minutes of rest between sets for big muscle groups like legs and chest, but 2-3 minutes for smaller muscle groups like arms and shoulders.

Tone: Encouraging and helpful 
"""

#Using llama3 for generating a response
openai.api_base = "https://api.groq.com/openai/v1"
openai.api_key = st.secrets["GROQ_API_KEY"]

client = OpenAI()

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": rag_prompt}],
    temperature=0.7
)

st.write(response.choices[0].message.content)


#local llama3 server
#uncomment to use local llama3 server
"""
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3", "prompt": rag_prompt, "stream":  False}
)

response_data = response.json()
if 'response' in response_data:
    print()
    st.write(response_data['response']) 
else:
    st.write("Error or unexpected response:", response_data) 
"""

