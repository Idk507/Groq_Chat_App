import streamlit as st
import os 
import groq 
from groq import Groq 
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory 
from langchain.prompts import PromptTemplate 
from dotenv import load_dotenv
from langchain_groq import ChatGroq 

load_dotenv()

groq_api_key = ''

def main():
    st.title("Groq Chat App")
    st.sidebar.title("Select an LLM")
    model = st.sidebar.selectbox('Select an LLm ',['Mistral-8x7b-32768','llama2-70b-4096'])
    conversational_memory_length = st.sidebar.slider('Conversational MEmory Length :',1,10,value=5)
    memory = ConversationBufferMemory(k=conversational_memory_length)
    user_question = st.text_area("Ask a question")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input': message['human']},{'output':message['AI']})
    
    groq_chat = ChatGroq(
        groq_api_key = groq_api_key,
        model_name = model
    )        
    
    conversation = ConversationChain(
        llm = groq_chat,
        memory = memory
    )
    
    if user_question != "":
        response = conversation(user_question)
        message = {'human': user_question, 'AI': response['response']}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:",response['response'])
        
if __name__ == "__main__":
    main()