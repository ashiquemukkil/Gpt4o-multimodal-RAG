from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
import uuid
import chromadb
from typing import List

from orchestrator.db.base import MultiModalDB



class MultiModalChromaDB:
    def __init__(self) -> None:
        self.chroma_client = chromadb.PersistentClient()
        super().__init__()

    def create_collection(self,collection_name):
        return self.chroma_client.get_or_create_collection(collection_name)

    def get_db(self,collection_name,embedding_function):
        return Chroma(
            client=self.chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )
    
    def get_multi_retriever(self,id_key,doc_store,vector_store):
        self.id_key = id_key
        self.multi_retriever = MultiVectorRetriever(
            vectorstore=vector_store,
            docstore=doc_store,
            id_key=id_key,
        )

    def add(self,raw_document:List, summary_documents:List,ids: List=None) -> None:
        if not ids:
            ids = [str(uuid.uuid4()) for _ in raw_document]

        summary_docs = [
            Document(page_content=s, metadata={self.id_key: ids[i]})
            for i, s in enumerate(summary_documents)
        ]

        self.multi_retriever.docstore.mset(list(zip(ids, raw_document)))
        self.multi_retriever.vectorstore.add_documents(summary_docs)

    def get(self,query:str,top_k:int = 5):
        return self.multi_retriever.get_relevant_documents(query, limit=top_k)

