import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os
import gc

def build_db():
    print("Loading TWCS dataset...")
    # Load only a subset to keep it feasible for local embedding
    # In a production environment, we would process the entire 3 million rows
    df = pd.read_csv("data/twcs/twcs.csv", nrows=100000)
    
    print("Parsing Q&A pairs...")
    # Brands are inbound=False
    agent_tweets = df[df['inbound'] == False].dropna(subset=['in_response_to_tweet_id'])
    customer_tweets = df[df['inbound'] == True]
    
    # Map tweet_id to text for fast lookup
    customer_dict = dict(zip(customer_tweets['tweet_id'], customer_tweets['text']))
    
    documents = []
    metadatas = []
    ids = []
    
    count = 0
    for _, agent_row in agent_tweets.iterrows():
        try:
            in_response_to = int(agent_row['in_response_to_tweet_id'])
        except ValueError:
            continue
            
        if in_response_to in customer_dict:
            customer_text = str(customer_dict[in_response_to])
            agent_text = str(agent_row['text'])
            brand = str(agent_row['author_id'])
            
            # The document to embed is the customer question
            documents.append(customer_text)
            metadatas.append({
                "agent_reply": agent_text,
                "brand": brand,
                "tweet_id": str(agent_row['tweet_id'])
            })
            ids.append(str(agent_row['tweet_id']))
            count += 1
            
            if count >= 10000:  # Limit to 10k pairs for fast local demo
                break
                
    print(f"Extracted {count} valid Q&A pairs.")
    
    # Free up memory
    del df
    del agent_tweets
    del customer_tweets
    del customer_dict
    gc.collect()
    
    print("Initializing ChromaDB...")
    db_path = os.path.join("data", "chroma_db")
    os.makedirs(db_path, exist_ok=True)
    
    client = chromadb.PersistentClient(path=db_path)
    
    # Use a lightweight local embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    collection = client.get_or_create_collection(name="twcs_support")
    
    print("Embedding and indexing documents (this may take a few minutes)...")
    
    # Batch insertion
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        # Embed locally
        embeddings = model.encode(batch_docs).tolist()
        
        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids
        )
        print(f"Indexed {min(i+batch_size, len(documents))} / {len(documents)}")
        
    print("Successfully built Vector Database!")

if __name__ == "__main__":
    build_db()
