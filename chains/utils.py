import yaml
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import load_prompt  # chat 형식이 아닌 그냥 prompt인경우 ChatPromptTemplate (x), PromptTemplate(o)


base_dir = os.path.dirname(__file__)  # 이 파일 기준


def get_chat_prompt_yaml(file_path):
    with open(file_path, "r", encoding='utf8') as f:
        yaml_content = yaml.safe_load(f)
        return [(message['role'], message['content']) for message in yaml_content['messages']]


def load_prompt_template(prompt_filename: str) -> ChatPromptTemplate:
    prompt_path = os.path.join(base_dir, "prompts", prompt_filename)
    prompt_yaml = get_chat_prompt_yaml(prompt_path)
    return ChatPromptTemplate.from_messages(prompt_yaml)