from .llm import get_OpenAILLM
from langchain_core.output_parsers import StrOutputParser
from .utils import load_prompt_template
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers.base import BaseOutputParser
from langchain_core.vectorstores.base import VectorStore
from rag.retriever.retriever import get_retrievered_documents
from typing import Optional


class VanillaChain:
    def __init__(
        self,
        prompt_file: str,
        llm: Optional[LanguageModelLike] = None,
        output_parser : Optional[BaseOutputParser] = None,
        openai_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = openai_api_key
        self.model_name = model_name
        self.prompt_file = prompt_file

        self.prompt = self._load_prompt()
        self.llm = llm or self._load_llm()
        self.parser = output_parser or StrOutputParser()

    def _load_prompt(self):
        return load_prompt_template(self.prompt_file)

    def _load_llm(self):
        return get_OpenAILLM(self.api_key, self.model_name)

    @property
    def chain(self):
        return self.prompt | self.llm | self.parser
    
    def __call__(self, input : str):
        return self.chain.invoke({"input": input})




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
    prompt = load_prompt_template("chat_context_answer_prompt.yaml")

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)   
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)



def get_vanilla_chain(openai_api_key, model_name):
    return VanillaChain(prompt_file="normal_prompt.yaml",
                        llm=get_OpenAILLM(openai_api_key, model_name),
                        output_parser=StrOutputParser())


def get_conversation_title_chain(openai_api_key, model_name):
    return VanillaChain(prompt_file="query_title_prompt.yaml",
                        llm=get_OpenAILLM(openai_api_key, model_name),
                        output_parser=StrOutputParser())