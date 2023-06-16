import numpy as np
from redis.commands.search.query import Query
import redis
import openai
import os
import logging

import azure.functions as func

# Redis connection details
redis_host = <>
redis_port = <>
redis_password = <>

openai.api_type = <>
openai.api_key = <>
openai.api_base = <>
openai.api_version = <>

def search_vectors(query_vector, client, top_k=5):
    base_query = "*=>[KNN 5 @embedding $vector AS vector_score]"
    query = Query(base_query).return_fields("Medicare", "vector_score").sort_by("vector_score").dialect(2)    

    try:
        results = client.ft("posts").search(query, query_params={"vector": query_vector})
    except Exception as e:
        print("Error calling Redis search: ", e)
        return None

    return results


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Connect to the Redis server
    conn = redis.Redis(host=redis_host, port=redis_port, 
                   password=redis_password, encoding='utf-8', decode_responses=True)

    if conn.ping():
        print("Connected to Redis")
    try:
        req_body = req.get_json()
        embedding = openai.Embedding.create(
                input=req_body["input"], engine="Embeddings-Test")
        query_vector = embedding["data"][0]["embedding"]
        query_vector = np.array(query_vector).astype(np.float32).tobytes()

        # Perform the similarity search
        print("Searching for similar posts...")
        result = search_vectors(query_vector, conn)

        results = []
        paragraphs = []
        print(f"Found {result.total} results")
        for i, post in enumerate(result.docs):
            results.extend(post.Medicare)
            paragraphs.extend(post.id)
        if (len(results) > 0):
            response = openai.Completion.create(
                model="text-davinci-003",
                max_tokens=1000,
                engine="GPT3",
                prompt=f"Using this information: \n\n{results}\n\n Generate a response for: \n\n{req_body['input']}"
            )
            return func.HttpResponse(response['choices'][0]['text'])
        else:
            return func.HttpResponse("No similar results found")
                
    except Exception as e:
        return func.HttpResponse(f"Unexpected Error: {e}")


    