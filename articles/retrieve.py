import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text 
import numpy as np
import os
import json
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import logging

os.environ['TFHUB_CACHE_DIR'] = 'tfhub_cache'

parser = argparse.ArgumentParser(description='Generate a prompt for a given paper and table.')
parser.add_argument('--max_tokens', type=int, default=50)
parser.add_argument('--min_similarity', type=float, default=0.55)
parser.add_argument('--topk', type=int, default=30)
args = parser.parse_args()



logging.basicConfig(level=logging.INFO)

hub_url = "https://www.kaggle.com/models/google/gtr/frameworks/TensorFlow2/variations/gtr-large/versions/1"
encoder = hub.KerasLayer(hub_url)

q1 = "What chemical is used in the polymer matrix?"
q2 = "What chemical is used in the polymer filler?"
q3 = "What is the filler mass % composition?"
q4 = "What is the filler volume % composition?"

queries = [q1, q2, q3, q4]

import spacy
nlp = spacy.load("en_core_web_sm") 

max_tokens = args.max_tokens
min_similarity = args.min_similarity
topk = args.topk


def rerank(csv_files, k=30):

    for csv_file in csv_files:
        retrieved = []
        
        with open(f'retrieved_chunks/{csv_file}', 'r') as file:
            rows = file.readlines()
            for row in rows:
                ques = row.split('|||')[0]
                id = row.split('|||')[1]
                similarity = row.split('|||')[2]
                text = row.split('|||')[3]
                retrieved.append((ques, id, similarity, text))

        doc_id = csv_file.split('.')[0]
        
        top_k_retrieved = {}
        for question in set([x[0] for x in retrieved]):
            top_k_retrieved[question] = sorted([x for x in retrieved if x[0] == question], key=lambda x: x[2], reverse=True)[:k]
        
        for question in top_k_retrieved:
            top_k_retrieved[question] = [x for x in top_k_retrieved[question]]

        all_chunks = []
        for question in top_k_retrieved:
            for chunk in top_k_retrieved[question]:
                chunk = (chunk[1], chunk[3])
                all_chunks.append(chunk)
        

        all_chunks = sorted(all_chunks, key=lambda x: int(x[0]))
        all_chunks = [x[1] for x in all_chunks]

        all_chunks = list(set(all_chunks))
        
        joined_chunks = " ".join(all_chunks)

        if not os.path.exists(f'ret_articles_top{k}'):
            os.makedirs(f'ret_articles_top{k}')

        with open(f'ret_articles_top{k}/{doc_id}.txt', 'w') as file:
            file.write(joined_chunks)


if __name__ == "__main__":
    all_files = os.listdir("all")
    

    for doc_id in all_files:

        logging.info(f"Processing {doc_id}")
        
        text = open(f"all/{doc_id}", "r").read()

        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        chunks = []
        chunk = ""
        for sentence in sentences:
            if len(chunk.split()) + len(sentence.split()) <= max_tokens:
                chunk += sentence
            else:
                chunks.append(chunk)
                chunk = sentence
        
        chunks = [chunk for chunk in chunks]

        chunk_embeddings = []
        
        # encode 5 chunks at a time
        for i in range(0, len(chunks), 5):
            chunk = chunks[i:i+5]
            logging.info(f"Processing chunk {i} to {i+5}")
            chunk_embedding = encoder(chunk)[0]
            chunk_embeddings.extend(chunk_embedding)

        query_embeddings = encoder(queries)[0]

        chunk_ids = np.arange(len(chunks))
        
        selected_chunks = {query: [] for query in queries} 

        for query, query_embedding in zip(queries, query_embeddings):

            for chunk_i, chunk_embedding in zip(chunk_ids, chunk_embeddings):
                similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
                
                chunk = chunks[chunk_i]

                if similarity >= min_similarity:
                    selected_chunks[query].append({
                        "chunk id": chunk_i,
                        "similarity": similarity,
                        "chunk": chunk
                    })
            
        # make retrieved_chunks if it doesn't exist
        if not os.path.exists("retrieved_chunks"):
            os.makedirs("retrieved_chunks")

        doc_id = doc_id.split(".")[0]
        with open(f"retrieved_chunks/{doc_id}.csv", "w") as f:
            for query, chunks in selected_chunks.items():
                for chunk in chunks:
                    f.write(f"{query}|||{chunk['chunk id']}|||{chunk['similarity']}|||{chunk['chunk']}\n")

    csv_files = [f for f in os.listdir('retrieved_chunks') if f.endswith('.csv')]
    rerank(csv_files, k=topk)
