import chromadb

from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



#embed_model = HuggingFaceEmbedding(model_name="/mnt/sdc/huggingface/model_hub/bge-m3")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def load_vector_store(collection_name):
    
    db = chromadb.PersistentClient(path="./chroma_db")

    # get collection
    chroma_collection = db.get_or_create_collection(collection_name)

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    return vector_store


def query_vector_store(vector_store, query_str, query_mode, similarity_top_k=1):

    query_embedding = embed_model.get_query_embedding(query_str)

    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=similarity_top_k, mode=query_mode
    )

    query_result = vector_store.query(vector_store_query)

    return query_result

if __name__ == "__main__":
    vector_store = load_vector_store("cot_examples")
    query_result = query_vector_store(vector_store, "caption this image with positive sentiment.", "default", similarity_top_k=1)
    print(query_result.nodes[0])
