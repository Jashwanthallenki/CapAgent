import chromadb

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def build_vector_store(documents_dir: str, collection_name: str):

    # load some documents
    documents = SimpleDirectoryReader(documents_dir).load_data()
    # initialize client, setting path to save data
    db = chromadb.PersistentClient(path="./chroma_db")

    # create collection
    chroma_collection = db.get_or_create_collection(collection_name)

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = HuggingFaceEmbedding(model_name="/mnt/sdc/huggingface/model_hub/bge-m3", max_length=8192)
    
    # create your index
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model, show_progress=True
    )

    # from IPython import embed; embed()

    return index

if __name__ == "__main__":
    build_vector_store(documents_dir="./data/cot_examples", collection_name="cot_examples")
