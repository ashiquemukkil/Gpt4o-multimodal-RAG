from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

from data_ingestion.gpt_4_o.config import OPENAI_KEY


import os
os.environ["OPENAI_API_KEY"] = OPENAI_KEY


def find_relevant_chunk(query,db,k=4):
    docs = db.similarity_search(query,k=k)
    return docs

def save_to_db(transcript):
    document = Document(page_content=transcript)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents([document])

    db = Chroma.from_documents(chunks, OpenAIEmbeddings())
    return db

