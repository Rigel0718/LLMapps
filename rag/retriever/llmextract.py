from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAI




def get_compression_retriever(compressor, retriever):
    return ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


if __name__ == '__main__':
    
    documents = TextLoader("state_of_the_union.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever()


    llm = OpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = get_compression_retriever(llm, retriever)
    compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
)
    print(compressed_docs)


