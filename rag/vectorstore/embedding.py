import os
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

OPENAI_EMBEDDING_MODEL = 'text-embedding-3-small'

def get_openai_embedding(openapi_key, model_name):
    return OpenAIEmbeddings(model=model_name, api_key=openapi_key)

huggingface_token = os.environ['HUGGINGFACEHUB_API_TOKEN']
DEFAULT_HUGGINGFACE_MODEL_NAME = 'intfloat/multilingual-e5-large-instruct'

def get_huggingface_endpoint_embeddings(huggingface_api_token=huggingface_token, model_name=DEFAULT_HUGGINGFACE_MODEL_NAME):
    return HuggingFaceEndpointEmbeddings(
        model=model_name,
        task='feature-extraction',
        huggingfacehub_api_token=huggingface_api_token,
    )