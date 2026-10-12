import sys
import os
import time
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neo4j.driver import Neo4jClient, get_neo4j
from config_loader import get_config

BATCH_SIZE = 128  # 100 nodes each batch
USE_LOCAL_MODEL = True # True for local embedding model

EMBEDDING_DIM = 768  # default value, will be replaced later

# Initialize model
if USE_LOCAL_MODEL:
    from sentence_transformers import SentenceTransformer
    config = get_config()
    local_model_path = config.get("local_emb_path", None)

    if local_model_path and os.path.exists(local_model_path):
        print(f"Loading local model: {local_model_path}")
        model = SentenceTransformer(local_model_path, local_files_only=True)
        # Try to get embedding dimension from model config
        EMBEDDING_DIM = model.get_sentence_embedding_dimension()
    else:
        raise ValueError("No local model/ Invalid local model!")

    print(f"Embedding initialized, embedding dimension: {EMBEDDING_DIM}")

    def get_embedding(text):
        return model.encode(text).tolist()
else:
    from openai import OpenAI
    client = OpenAI(api_key="sk-...", base_url="...")
    EMBEDDING_DIM = 1536

    def get_embedding(text):
        resp = client.embeddings.create(input=text, model="text-embedding-3-small")
        return resp.data[0].embedding

def main():
    client = Neo4jClient()
    print("Loading to Neo4j Database")

    count_query = "MATCH (n:Entity) WHERE n.embedding IS NULL RETURN count(n) as total"
    result = client.query(count_query)
    
    if result and isinstance(result, list) and len(result) > 0:
        total = result[0].get('total', 0)
    else:
        total = 0
        
    print(f"Find {total} nodes that need to Embed")

    if total == 0:
        print("All nodes embedded")
        return

    # batch process
    pbar = tqdm(total=total)
    
    while True:
        # get a batch of un-processed nodes
        fetch_query = """
        MATCH (n:Entity) 
        WHERE n.embedding IS NULL 
        RETURN elementId(n) as id, n.name as text 
        LIMIT $limit
        """
        nodes = client.query(fetch_query, {"limit": BATCH_SIZE})
        
        if not nodes:
            break

        updates = []
        for node in nodes:
            text = node.get('text', '')
            # avoid void text error
            if not text or len(str(text).strip()) == 0:
                vector = [0.0] * EMBEDDING_DIM
            else:
                vector = get_embedding(str(text))
            
            updates.append({"id": node['id'], "vector": vector})

        # batch rewrite
        update_query = """
        UNWIND $updates AS row
        MATCH (n) WHERE elementId(n) = row.id
        SET n.embedding = row.vector
        """
        
        client.query(update_query, {"updates": updates})
        
        pbar.update(len(nodes))

    pbar.close()
    print("All nodes' embedding injected!")

    print(f"Creating vector index (dimension: {EMBEDDING_DIM})...")
    create_index_query = f"""
    CREATE VECTOR INDEX node_embedding_index IF NOT EXISTS
    FOR (n:Entity) ON (n.embedding)
    OPTIONS {{indexConfig: {{
     `vector.dimensions`: {EMBEDDING_DIM},
     `vector.similarity_function`: 'cosine'
    }}}}
    """
    try:
        client.query(create_index_query)
        print("Creates/Exists 'node_embedding_index'")
    except Exception as e:
        print(f" Warning appears when creating 'node_embedding_index' :{e}")

if __name__ == "__main__":
    main()