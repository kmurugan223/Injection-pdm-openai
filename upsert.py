import os
from dotenv import load_dotenv
from pinecone import Pinecone
import uuid
from openai import OpenAI

# Load environment variables
load_dotenv()

# Get API keys from .env file
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# Set the index name - replace with your actual index name
INDEX_NAME = "injection-pdm"  # Replace with your index name

# Connect to the index
index = pc.Index(INDEX_NAME)

# Define some CS concepts to upsert
cs_concepts = [
    {
        "id": str(uuid.uuid4()),
        "content": "Artificial Intelligence (AI) is a field of computer science focused on creating systems that can perform tasks that typically require human intelligence.",
        "metadata": {
            "topic": "AI",
            "subtopic": "general",
            "difficulty": "beginner"
        }
    },
    {
        "id": str(uuid.uuid4()),
        "content": "Machine Learning is a subset of AI that focuses on developing algorithms that can learn from and make predictions on data.",
        "metadata": {
            "topic": "AI",
            "subtopic": "machine learning",
            "difficulty": "beginner"
        }
    },
    {
        "id": str(uuid.uuid4()),
        "content": "Retrieval-Augmented Generation (RAG) is an AI framework that combines retrieval-based methods with generative models to enhance text generation with external knowledge.",
        "metadata": {
            "topic": "RAG",
            "subtopic": "general",
            "difficulty": "intermediate"
        }
    },
    {
        "id": str(uuid.uuid4()),
        "content": "RAG systems typically involve a retrieval component that finds relevant documents and a generative component that produces responses based on the retrieved information.",
        "metadata": {
            "topic": "RAG",
            "subtopic": "architecture",
            "difficulty": "intermediate"
        }
    },
    {
        "id": str(uuid.uuid4()),
        "content": "Large Language Models (LLMs) are AI systems trained on vast amounts of text data that can generate human-like text and perform various language tasks.",
        "metadata": {
            "topic": "AI",
            "subtopic": "LLMs",
            "difficulty": "intermediate"
        }
    }
]

# Function to get embeddings from OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding  # This returns a 1536-dimensional vector

# Convert content to embeddings and prepare for upsert
vectors_to_upsert = []

for concept in cs_concepts:
    # Generate embedding for the concept content (1536 dimensions)
    embedding = get_embedding(concept["content"])
    
    # Create vector record
    vector = {
        "id": concept["id"],
        "values": embedding,
        "metadata": {
            **concept["metadata"],
            "text": concept["content"]  # Store the original text in metadata for retrieval
        }
    }
    
    vectors_to_upsert.append(vector)

# Upsert vectors to Pinecone (batch size of 100 at most)
batch_size = 100
for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i+batch_size]
    upsert_response = index.upsert(vectors=batch)
    print(f"Upserted batch {i//batch_size + 1}, response: {upsert_response}")

print(f"Upserted {len(vectors_to_upsert)} total vectors to Pinecone")

# Optional: Test retrieval by querying for "RAG"
query = "What is RAG?"
query_embedding = get_embedding(query)

# Search the index
search_results = index.query(
    vector=query_embedding,
    top_k=2,
    include_metadata=True
)

print("\nSearch results for 'What is RAG?':")
for result in search_results['matches']:
    print(f"Score: {result['score']}")
    print(f"Content: {result['metadata'].get('text', 'No text available')}")
    print(f"Metadata: {result['metadata']}")
    print("---")