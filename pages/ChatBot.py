import streamlit as st
from util import get_text, get_text_chunks, get_vectorstore, get_conversation_chain

MODEL = ['model1', 'model2', 'model3']

def main():
    st.set_page_config(
        page_icon=':books:',
        page_title='ChatQA'
    )
    st.title("💬 Chatbot")

    st.write("Hello, world!")

    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None

    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = None

    with st.sidebar:
        upload_files = st.file_uploader('upload your files', type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        st.text_input("OpenAI API Key", key="openai_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        st.selectbox("🤖 Select a Model", options=MODEL, key = 'model')
        process = st.button('process')
        

    # if process:
    #     if not openai_api_key:
    #         st.info('Please add your OpenAI API KEY to continue')
    #         st.stop()
    #     text_files = get_text(upload_files)
    #     text_chunks = get_text_chunks(text_files)
    #     vectors = get_vectorstore(text_chunks)

    #     st.session_state.conversation_chain = get_conversation_chain(vectors)   

    # session state initialize
    if 'messages' not in st.session_state:
        # st.session_state.messages = [{"role": "assistant", "content": "Please, input your OpenAI key on sidebar and upload your own files?"}]
        # st.chat_message('assistant').write('Please, input your OpenAI key on sidebar and upload your own files?')
        st.session_state.messages = [{"role": "assistant", "content": "Hello, How can I help you?"}]


    
    # st.session_state.messages = [{"role": "assistant", "content": "Hello, How can I help you?"}]

    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    if query := st.chat_input('Please enter your Question'):
        # if not openai_api_key:
        #     st.info("Please add your OpenAI API key to continue.")
        #     st.stop()
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message('user').write(query)
        msg = 'respons'
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

if __name__ == '__main__':
    main()
