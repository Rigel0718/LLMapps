from langchain_openai import ChatOpenAI


def get_OpenAILLM(openai_api_key, model_name):
    return ChatOpenAI(
        api_key=openai_api_key, 
        model=model_name,
        temperature=0.3,
        streaming=True )