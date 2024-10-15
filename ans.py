from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from datasets import Dataset
import torch
import faiss

# Initialize Question Encoder and Tokenizer
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

# Initialize Context (Passage) Encoder and Tokenizer
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Example corpus of documents (knowledge base)
corpus = [
    "Roy is a common first name in many countries.",
    "There are thousands of people named Roy across the world.",
    "Roy is a name often associated with royalty and nobility.",
    "The name Roy has origins in both English and French languages."
]

# Encode the corpus documents
def encode_corpus(corpus):
    inputs = context_tokenizer(corpus, padding=True, truncation=True, return_tensors="pt")
    embeddings = context_encoder(**inputs).pooler_output
    return embeddings

# Function to retrieve the best matching document
def retrieve_answer(question, corpus, corpus_embeddings):
    # Encode the question
    question_inputs = question_tokenizer(question, return_tensors="pt")
    question_embedding = question_encoder(**question_inputs).pooler_output
    
    # Use FAISS to find the closest document in the corpus
    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])  # Use Inner Product (IP) for similarity
    faiss.normalize_L2(corpus_embeddings.numpy())  # Normalize corpus embeddings for cosine similarity
    index.add(corpus_embeddings.numpy())  # Add the corpus embeddings to the index
    
    # Search for the nearest document
    faiss.normalize_L2(question_embedding.detach().numpy())
    D, I = index.search(question_embedding.detach().numpy(), k=1)  # k=1 for top match

    # Retrieve and return the top matching document
    return corpus[I[0][0]], D[0][0]

# Encode the corpus
corpus_embeddings = encode_corpus(corpus)

# Example question
question = "how many people are there with first name roy?"

# Retrieve the best matching document and print the result
best_match, score = retrieve_answer(question, corpus, corpus_embeddings)
print(f"Best matching document: {best_match} (score: {score})")
