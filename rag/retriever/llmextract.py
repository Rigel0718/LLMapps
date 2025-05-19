from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain.retrievers.document_compressors import LLMChainFilter, EmbeddingsFilter, DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
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
    from langchain_openai import OpenAIEmbeddings
    documents = TextLoader("state_of_the_union.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever()


    llm = OpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = get_compression_retriever(compressor, retriever)
    compressed_docs = compression_retriever.invoke(
    "What did the president say about Ketanji Jackson Brown"
    )
    

    embeddings = OpenAIEmbeddings()
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    get_compression_retriever(compressor=embeddings_filter, )
    print(compressed_docs)


    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    # EmbeddingsRedundantFilter : similarity_threshold를 0.95로 잡아서 0.95보다 높은 문서는 같은문서로 생각하고 중복된 context를 없애는 filter
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings) 
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )

    compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor, base_retriever=retriever
    )
