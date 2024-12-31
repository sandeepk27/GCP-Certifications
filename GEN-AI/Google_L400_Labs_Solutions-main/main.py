import os
import json
import logging
import google.cloud.logging
from flask import Flask, render_template, request

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from langchain_google_vertexai import VertexAIEmbeddings

# Configure Cloud Logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# Read application variables from the config fle
BOTNAME = "FreshBot"
SUBTITLE = "Your Friendly Restaurant Safety Expert"

app = Flask(__name__)

# Initializing the Firebase client
db = firestore.Client()

# Firestore Collection Reference
collection = db.collection('food-safety')

# TODO: Instantiate an embedding model here
embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")

# TODO: Instantiate a Generative AI model here
gen_model = GenerativeModel(model_name="gemini-1.5-pro-001")

# TODO: Implement this function to return relevant context
# from your vector database
def search_vector_database(query: str):
    query_embedding = embedding_model.embed_query(query)
    vector_query = collection.find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.EUCLIDEAN,
        limit=5,
    )
    docs = vector_query.stream()
    context = " ".join([result.to_dict()['content'] for result in docs])


    # Don't delete this logging statement.
    logging.info(
        context, extra={"labels": {"service": "cymbal-service", "component": "context"}}
    )
    return context

# TODO: Implement this function to pass Gemini the context data,
# generate a response, and return the response text.
# Function to Send Query and Context to Gemini and Get the Response
def ask_gemini(question):
    # Create a prompt template with context instructions
    prompt_template = "Using the context provided below, answer the following question:\nContext: {context}\nQuestion: {question}\nAnswer:"
    
    # Retrieve context for the question using the search_vector_database function
    context = search_vector_database(question)
    
    # Format the prompt with the question and retrieved context
    formatted_prompt = prompt_template.format(context=context, question=question)
    
    # Define the generation configuration for the Gemini model
    generation_config = GenerationConfig(
        temperature=0.7,  # Adjust temperature as needed
        max_output_tokens=256,  # Maximum tokens in the response
        response_mime_type="application/json",  # MIME type of response
    )
    
    # Define the contents parameter with the prompt text
    contents = [
        {
            "role": "user",
            "parts": [{"text": formatted_prompt}]
        }
    ]
    
    # Call the generate_content function with the defined parameters
    response = gen_model.generate_content(
        contents=contents,
        generation_config=generation_config
    )
    
    # Parse the JSON response to extract the answer field
    response_text = response.text if response else "{}"  # Default to empty JSON if no response
    try:
        response_json = json.loads(response_text)  # Parse the JSON string into a dictionary
        answer = response_json.get("answer", "No answer found.")  # Get the "answer" field
    except json.JSONDecodeError:
        answer = "Error: Unable to parse response."

    return answer

# The Home page route
@app.route("/", methods=["POST", "GET"])
def main():

    # The user clicked on a link to the Home page
    # They haven't yet submitted the form
    if request.method == "GET":
        question = ""
        answer = "Hi, I'm FreshBot, what can I do for you?"

    # The user asked a question and submitted the form
    # The request.method would equal 'POST'
    else:
        question = request.form["input"]
        # Do not delete this logging statement.
        logging.info(
            question,
            extra={"labels": {"service": "cymbal-service", "component": "question"}},
        )
        
        # Ask Gemini to answer the question using the data
        # from the database
        answer = ask_gemini(question)

    # Do not delete this logging statement.
    logging.info(
        answer, extra={"labels": {"service": "cymbal-service", "component": "answer"}}
    )
    print("Answer: " + answer)

    # Display the home page with the required variables set
    config = {
        "title": BOTNAME,
        "subtitle": SUBTITLE,
        "botname": BOTNAME,
        "message": answer,
        "input": question,
    }

    return render_template("index.html", config=config)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
