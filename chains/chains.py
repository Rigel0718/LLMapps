from llm import get_OpenAILLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils import get_chat_prompt_yaml


def get_vanilla_chain(openai_api_key, model_name):
    prompt_filepath = '.prompts/normal_prompt.yaml'
    prompt = ChatPromptTemplate(get_chat_prompt_yaml(prompt_filepath))

    return prompt | get_OpenAILLM(openai_api_key, model_name) | StrOutputParser()