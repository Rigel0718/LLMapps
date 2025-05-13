from .llm import get_OpenAILLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .utils import get_chat_prompt_yaml, load_prompt_template
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.language_models import LanguageModelLike
from langchain_core.vectorstores.base import VectorStore
from rag.retriever import get_retrievered_documents
import os

base_dir = os.path.dirname(__file__)


def get_vanilla_chain(openai_api_key, model_name):
    prompt = load_prompt_template("normal_prompt.yaml")
    return prompt | get_OpenAILLM(openai_api_key, model_name) | StrOutputParser()



def get_conversational_rag_chain(vectorstore : VectorStore, openai_api_key, model_name):
    llm = get_OpenAILLM(openai_api_key, model_name)
    retriever_chain = get_retrievered_documents(vectorstore, llm)
    '''
    prompt 
    system : LLM의 역할 정의
    context : 검색된 문서  (context인 이유는 create_stuff_documents_chain 함수의 document variable parameter의 기본값이 context이기 때문이다.)
    messages : 기존 대화의 흐름 정보
    input : user의 쿼리
    '''
    prompt = load_prompt_template("basic_prompt.yaml")

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)   
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)



def get_conversation_title_chain(openai_api_key, model_name):
    prompt = load_prompt_template("query_title_prompt.yaml")
    return prompt | get_OpenAILLM(openai_api_key, model_name) | StrOutputParser()