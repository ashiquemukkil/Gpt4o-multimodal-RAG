from orchestrator.db.chroma import MultiModalChromaDB
from data_ingestion.gpt_4_o.config import OPENAI_KEY
from data_ingestion.utils import encode_image

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.storage import InMemoryStore

import pandas as pd
import uuid


if __name__ == "__main__":
    texts_with_metadata = pd.read_csv("text_metadata_df_final.csv")
    images_with_metadata = pd.read_csv("image_metadata_df_final.csv")
    texts = images_with_metadata["img_desc"].to_list()
    images = images_with_metadata["img_path"].to_list()
    
    retriever_multi_vector_img = MultiModalChromaDB()
    embedd = OpenAIEmbeddings(model="text-embedding-3-large",openai_api_key=OPENAI_KEY)

    vector_store = retriever_multi_vector_img.get_db(collection_name="mm_rag_vect",embedding_function=embedd)
    # doc_store = retriever_multi_vector_img.get_db(collection_name="mm_rag_store",embedding_function=embedd)
    doc_store = InMemoryStore()
    mm_index = retriever_multi_vector_img.get_multi_retriever("doc_id",doc_store,vector_store)
    
    retriever_multi_vector_img.add(images,texts)

    while True:
        user_input = input("to exit entre Y or type usser query")
        if user_input == "Y":
            exit()

        print(retriever_multi_vector_img.get(user_input,top_k=3))
        