import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os
import gc

def build_multi_brand_db():
    print("Loading dataset for Vector DB construction...")
    documents = []
    metadatas = []
    ids = []
    brand_counts = {}

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_path = os.path.join(base_dir, "data", "train_retrieval_corpus.csv")
    twcs_path = os.path.join(base_dir, "data", "twcs", "twcs.csv")
    db_path = os.path.join(base_dir, "data", "chroma_db")


    if os.path.exists(corpus_path):
        print(f"Loading pre-extracted retrieval corpus from {corpus_path}...")
        cdf = pd.read_csv(corpus_path)
        for idx, row in cdf.iterrows():
            c_text = str(row.get("customer_text", "")).strip()
            a_text = str(row.get("agent_text", "")).strip()
            brand = str(row.get("brand", "CompanySupport")).strip()
            if not c_text or not a_text or c_text.lower() == "nan":
                continue
            
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
            if brand_counts[brand] > 1500:
                continue
            
            documents.append(c_text)
            metadatas.append({
                "agent_reply": a_text,
                "brand": brand,
                "tweet_id": str(row.get("agent_tweet_id", idx))
            })
            ids.append(f"{brand}_{idx}")
        print(f"Loaded {len(documents)} Q&A exemplars from {corpus_path}.")
    elif os.path.exists(twcs_path):
        print(f"Parsing raw TWCS dataset from {twcs_path}...")
        df = pd.read_csv(twcs_path, nrows=250000)
        agent_tweets = df[df['inbound'] == False].dropna(subset=['in_response_to_tweet_id'])
        customer_tweets = df[df['inbound'] == True]
        customer_dict = dict(zip(customer_tweets['tweet_id'], customer_tweets['text']))
        
        for idx, agent_row in agent_tweets.iterrows():
            try:
                in_response_to = int(agent_row['in_response_to_tweet_id'])
            except ValueError:
                continue
                
            if in_response_to in customer_dict:
                customer_text = str(customer_dict[in_response_to])
                agent_text = str(agent_row['text'])
                brand = str(agent_row['author_id'])
                
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
                if brand_counts[brand] > 1500:
                    continue
                    
                documents.append(customer_text)
                metadatas.append({
                    "agent_reply": agent_text,
                    "brand": brand,
                    "tweet_id": str(agent_row['tweet_id'])
                })
                ids.append(f"{brand}_{agent_row['tweet_id']}")
        del df, agent_tweets, customer_tweets, customer_dict
        gc.collect()
    else:
        print("Error: Neither train_retrieval_corpus.csv nor twcs.csv found.")
        return

    os.makedirs(db_path, exist_ok=True)

    
    client = chromadb.PersistentClient(path=db_path)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Recreate TWCS multi-brand collection
    try:
        client.delete_collection("twcs_support")
    except Exception:
        pass
    collection = client.create_collection(name="twcs_support")
    
    print("Embedding and indexing multi-brand TWCS documents...")
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        embeddings = model.encode(batch_docs).tolist()
        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids
        )
        print(f"TWCS Indexed {min(i+batch_size, len(documents))} / {len(documents)}")

    # Index Banking77 for secondary intent reference if present
    banking_train_path = "data/banking77/train.csv"
    if os.path.exists(banking_train_path):
        print("Indexing Banking77 intent reference dataset...")
        b_df = pd.read_csv(banking_train_path)
        b_docs = b_df['text'].astype(str).tolist()
        b_meta = [{"intent_label": str(lbl)} for lbl in b_df['label']]
        b_ids = [f"banking77_{i}" for i in range(len(b_docs))]
        
        try:
            client.delete_collection("banking77_intents")
        except Exception:
            pass
        b_collection = client.create_collection(name="banking77_intents")
        
        for i in range(0, len(b_docs), batch_size):
            bd = b_docs[i:i+batch_size]
            bm = b_meta[i:i+batch_size]
            bi = b_ids[i:i+batch_size]
            be = model.encode(bd).tolist()
            b_collection.add(documents=bd, embeddings=be, metadatas=bm, ids=bi)
        print("Successfully indexed Banking77 intents!")

    print("Multi-Brand Vector Database construction complete!")

if __name__ == "__main__":
    build_multi_brand_db()
