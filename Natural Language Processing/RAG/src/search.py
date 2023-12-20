import os

from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
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



def query(search_term):
    
    try:
        collection = get_mongodb_collection()
    
        embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get('OPENAI_RAG_KEY'))

        _vectorStoreMongo = MongoDBAtlasVectorSearch(collection=collection, embedding=embeddings)

        docs = _vectorStoreMongo.similarity_search(search_term, k=1) 
        out = docs[0].page_content

        retriever = _vectorStoreMongo.as_retriever()

        llm = OpenAI(openai_api_key=os.environ.get('OPENAI_RAG_KEY'), temperature=0)

        qa_stuff = RetrievalQA.from_chain_type(llm=llm, chain_type = "stuff",
        retriever = retriever,verbose = True)
        retriever_out = qa_stuff.run(search_term)
    
        return out, retriever_out

        print(f"Execution finished...")
    except Exception as ex:
        print(ex)

# What questions did Shashank ask? What were Krishna's answer? Please summarize them in bullets