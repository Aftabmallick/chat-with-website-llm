import streamlit as st
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma,Pinecone
import pinecone
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
#load_dotenv()
import os



def get_vectorstore_from_url(url,opkey,index_name,delete=0):
    #get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()

    #split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks=text_splitter.split_documents(document)
    #create a vector store from the chunks
    vector_store = Pinecone.from_documents(document_chunks,OpenAIEmbeddings(openai_api_key=opkey),index_name=index_name)
    if delete ==1:
        vector_store.delete(delete_all=True)
    return vector_store
def get_context_retriever_chain(vector_store,opkey):
    llm = ChatOpenAI(openai_api_key=opkey)

    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
        ("user","Given the above conversation, generate a search query to look up in order to get information relevent to this conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm,retriever,prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain,opkey):
    llm = ChatOpenAI(openai_api_key=opkey)
    prompt = ChatPromptTemplate.from_messages([
        ("system","Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
    ])
    stuff_document_chain = create_stuff_documents_chain(llm,prompt)

    return create_retrieval_chain(retriever_chain,stuff_document_chain)

def get_response(user_input):
    #create conversation chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store,openai_api_key)
    convesation_rag_chain=get_conversational_rag_chain(retriever_chain,openai_api_key)
    response = convesation_rag_chain.invoke({
            "chat_history":st.session_state.chat_history,
            "input":user_query
            })
    return response['answer']

def pinecone_init(openai_api_key,pinecone_api_key,pinecone_environment,index_name):
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_environment
    )
    index_name=index_name

#app config
st.set_page_config(page_title="Chat With Websites",page_icon="")

st.title("Chat With Website")

#side bar
with st.sidebar:
    with st.form("my"):
        st.header("Settings")
        openai_api_key=st.text_input("OpenAi Api Key")
        pinecone_api_key=st.text_input("Pinecone Api Key")
        pinecone_environment=st.text_input("Pinecone Environment Name")
        index_name=st.text_input("Pinecone Index Name")
        submitted = st.form_submit_button("Submit")
        if submitted:
            pinecone_init(openai_api_key,pinecone_api_key,pinecone_environment,index_name)
with st.sidebar:
    website_url=st.text_input("Website URL")
    delete=st.button("Delete Indexs")
    if delete:
        get_vectorstore_from_url(website_url,openai_api_key,index_name,delete=1)

if website_url is None or website_url=="":
    st.info("Please enter all details")
else:
    #session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[
            AIMessage(content = "Hello I am a bot, How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url,openai_api_key,index_name)
        
    
    #user input
    user_query=st.chat_input("Type your message here..")
    if user_query is not None and user_query !="":
        response = get_response(user_query)
        
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

        

    #conversation
    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
