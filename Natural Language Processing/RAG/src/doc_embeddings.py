import os

from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms.openai import OpenAI
from langchain.chains import RetrievalQA


def connect_to_mongodb():
    print('Executing connect_to_mongodb...')
    return MongoClient(os.environ.get("MONGO_HOST"))

def get_mongodb_collection():
    print('Executing get_mongodb_collection...')
    client = connect_to_mongodb()
    db_name = "langchain_demo"
    collection_name = "text_blobs"
    return client[db_name][collection_name]
    
def load_documents():
    print('Executing load_documents...')
    loader = DirectoryLoader('../sample_files', glob="./*.txt", show_progress=True)
    return loader.load()

# print(documents)
def run_workflow():
    try:
        documents = load_documents()
        collection = get_mongodb_collection()
        embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get('OPENAI_RAG_KEY'))
        _vectorStoreMongo = MongoDBAtlasVectorSearch.from_documents(
        documents, embeddings, collection=collection) 
        print(f"Execution finished...")
        # return 0
    except Exception as ex:
        print(ex)


# run_workflow()