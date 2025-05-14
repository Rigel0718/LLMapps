import os

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

def get_openai_embedding(openapi_key, model_name):
    return OpenAIEmbeddings(model=model_name, api_key=openapi_key)

huggingface_token = os.environ['HUGGINGFACEHUB_API_TOKEN']
default_huggingface_model_name = 'intfloat/multilingual-e5-large-instruct'

def get_huggingface_endpoint_embeddings(huggingface_api_token=huggingface_token, model_name=default_huggingface_model_name):
    return HuggingFaceEndpointEmbeddings(
        model=model_name,
        task='feature-extraction',
        huggingfacehub_api_token=huggingface_api_token,
    )