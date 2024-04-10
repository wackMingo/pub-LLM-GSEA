import os
import openai
import sys
sys.path.append('../..')
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_openai import OpenAIEmbeddings
import numpy as np
from langchain_community.vectorstores import Chroma, DocArrayInMemorySearch

from langchain_community.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
#set API key for OpenAI in OS

def pdf():
#loading PDF
    loader = PyPDFLoader("2210.10341.pdf")
    pages = loader.load()
    len(pages)
    page = pages[0]
    print(page.page_content[0:500])
    print(page.metadata)

def yt():
    url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
    save_dir="docs/youtube/"
    loader = GenericLoader(
        YoutubeAudioLoader([url],save_dir),
        OpenAIWhisperParser()
    )
    docs = loader.load()
    docs[0].page_content[0:500]

def url():
    loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/37signals-is-you.md")
    docs = loader.load()
    print(docs[0].page_content[:500])

def split():
    chunk_size =26
    chunk_overlap = 4
    r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap)
    c_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)

    text1 = 'abcdefghijklmnopqrstuvwxyz'
    print(r_splitter.split_text(text1))
    text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
    print(r_splitter.split_text(text2))
    text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
    print(r_splitter.split_text(text3))
    print(c_splitter.split_text(text3))

    c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator = ' ')
    print(c_splitter.split_text(text3))

def split2():
    some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""
    len(some_text)
    c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator = ' ')
    r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0, 
    separators=["\n\n", "\n", " ", ""])
    #print(c_splitter.split_text(some_text))
    #print(r_splitter.split_text(some_text))

    r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", " ", ""])
    #print(r_splitter.split_text(some_text))

    r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=155,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )",  " ", " ", ""])
    #print(r_splitter.split_text(some_text))

    loader = PyPDFLoader("2210.10341.pdf")
    pages = loader.load()
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len)
    docs = text_splitter.split_documents(pages)
    #print(len(docs))
    #print(len(pages))
    return pages

def split_token(pages):
    text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
    text1 = "foo bar bazzyfoo"
    text_splitter.split_text(text1)
    text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)

def vectors():
    loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("2210.10341.pdf"),
    PyPDFLoader("2309.04019.pdf"),
    PyPDFLoader("2403.12038.pdf"),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150)
    splits = text_splitter.split_documents(docs)
    return splits

def embeds():
    embedding = OpenAIEmbeddings()
    sentence1 = "i like dogs"
    sentence2 = "i like canines"
    sentence3 = "the weather is ugly outside"
    embedding1 = embedding.embed_query(sentence1)
    embedding2 = embedding.embed_query(sentence2)
    embedding3 = embedding.embed_query(sentence3)
    print(np.dot(embedding1, embedding2))

def db(splits):
    embedding = OpenAIEmbeddings()
    persist_directory = 'docs/chroma/'

    vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory)
    question = "is there an email i can ask for help"
    docs = vectordb.similarity_search(question,k=3)
    print(len(docs))
    print(docs[0].page_content)
    vectordb.persist() 

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

def retr():
    persist_directory = 'docs/chroma/'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding)
    print(vectordb._collection.count())

    texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",]
    smalldb = Chroma.from_texts(texts, embedding=embedding)
    question = "Tell me about all-white mushrooms with large fruiting bodies"
    #print(smalldb.similarity_search(question, k=2))
    
    #max marginal relevance search for diversity and relevance
    #print(smalldb.max_marginal_relevance_search(question,k=2, fetch_k=3))
    question = "what did they say about genes?"
    #docs_ss = vectordb.similarity_search(question,k=3)
    #print(docs_ss[1].page_content[:100])
    #print(docs_ss[0].page_content[:100])

    #docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)
    """
    print(docs_mmr[0].page_content[:100])
    print("-")
    print(docs_mmr[1].page_content[:100])
    print("-")
    print(docs_mmr[2].page_content[:100])
    """

    #search in specific pdf
    """
    docs = vectordb.similarity_search(
    question,
    k=3,
    filter={"source":"2210.10341.pdf"})
    for d in docs:
        print(d.metadata)
        print(d)
    """

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The lecture the chunk is from, should be one of `2210.10341.pdf`, `2309.04019.pdf`, or `2403.12038.pdf`",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="The page from the paper",
            type="integer",
        ),
    ]

    document_content_description = "Paper notes"
    llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectordb,
        document_content_description,
        metadata_field_info,
        verbose=True)
    docs = retriever.get_relevant_documents(question)
    #for d in docs:
    #    print(d.metadata)
    #    print(d.page_content)


    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever())

    compressed_docs = compression_retriever.get_relevant_documents(question)
    """
    THIS WORKS GOOD!
    """
    pretty_print_docs(compressed_docs)

    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type = "mmr"))
    compressed_docs = compression_retriever.get_relevant_documents(question)
    pretty_print_docs(compressed_docs)

def qa():
    llm_name = "gpt-3.5-turbo"
    persist_directory = 'docs/chroma/'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    #print(vectordb._collection.count())

    question = "What are major topics for these papers?"
    docs = vectordb.similarity_search(question,k=3)
    len(docs) #774

    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever()
    )
    result = qa_chain({"query": question})
    #print(result["result"])

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    question = "Is evaluating large language models for gene set discovery a topic?"
    result = qa_chain({"query": question})
    #print(result["result"])
    #more info as to where the info came from
    #print(result["source_documents"][0])

    #if many documents, do every document to LLM and combine every answer
    qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce")
    result = qa_chain_mr({"query": question})
    #print(result["result"])

    #refine begins with one prompt, improves on it
    qa_chain_rf = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="refine")
    result = qa_chain_rf({"query": question})
    print(result["result"])

    #no conversational history (yet?)

def chat():
    persist_directory = 'docs/chroma/'
    llm_name = "gpt-3.5-turbo"
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    question = "What are major topics for this class?"
    llm = ChatOpenAI(model_name=llm_name, temperature=0)
    #llm.invoke("Hello world!")

    # Build prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    # Run chain
    question = "Is LLM's performance on gene set discovery a paper topic?"
    qa_chain = RetrievalQA.from_chain_type(llm,
                                        retriever=vectordb.as_retriever(),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


    #result = qa_chain({"query": question})
    #print(result["result"])
    #print(result)

    memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True)

    from langchain.chains import ConversationalRetrievalChain
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    print(result['answer'])
    #with memory!
    question2 = "What were the conclusions of this paper?"
    result = qa({"question": question2})
    print(result['answer'])

    #there are lots of different options for memory and question types

def load_db(file, chain_type, k):
    llm_name = "gpt-3.5-turbo"

    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qas = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name=llm_name, temperature=0), 
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qas


def main():
    #pdf()
    #yt()
    #url()

    #Splitting
    #split()
    #pages = split2()
    #split_token(pages)

    #Vectors and embeddings
    #splits = vectors()
    #embeds()
    #db(splits)

    #Retrieval
    #retr()
    #qa()
    #chat()
    file = "2210.10341.pdf"
    chain_type = "map_reduce"
    k=1
    question = "What are major topics for these papers?"
    qas = load_db(file, chain_type, k)
    result = qas({"question": question})
    print(result['answer'])


main()