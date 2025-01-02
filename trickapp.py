import os
import streamlit as st
from streamlit import secrets
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain.tools import Tool
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

st.set_page_config(page_title='AI DRK📜',page_icon='🦅')
st.title(body='What can I help with?')

# os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_API_KEY']=st.secrets['LANGCHAIN_API_KEY']

os.environ['LANGCHAIN_TRACING_V2']='true'

# os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')
os.environ['HF_TOKEN']=st.secrets['HF_TOKEN']

os.environ['LANGCHAIN_PROJECT']="Mutli AI"

# GROQ_API_KEY=os.getenv('GROQ_API_KEY')
GROQ_API_KEY=st.secrets['GROQ_API_KEY']

session_id=st.text_input('Session Id',value='default session')
splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=250)
HF_Embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

prompts_dict = {
    "PDF User Query": (
        "Using the user input provided as context: {context}, retrieve and answer the query from the content of the uploaded PDF(s), in minimum possible words. "
        "Ensure that the response is accurate, concise, and directly addresses the query. "
        "If the query cannot be answered using the PDF(s), respond with 'I don't know'."
    ),
    "Prompt Generator": (
        "Using the keywords provided in the context: {context}, generate a concise and clear prompt that can guide a task or system to achieve a specific goal. "
        "The generated prompt should focus solely on the keywords provided, without solving the task itself."    ),
"Movie Similarity Search": (
    "Using the keywords provided in the context: {text}, suggest movies that are relevant based on their plot, theme, or similarity to the mentioned movie name(s). "
    "If the context includes a movie name, suggest movies with a similar plot, genre, or style. "
    "For generic keywords, recommend movies that align with the themes or ideas represented by the keywords. "
    "Provide a brief explanation of how each recommended movie relates to the input context."
),
    "Essay Writer": (
    "Create a professional and unique essay based on the provided topic: {context}. "
    "Use the information from the uploaded PDF(s), but rephrase and summarize it in your own words. "
    "You may include tables and other relevant structures to present the information clearly and concisely. "
    "Ensure that the essay is well-structured, provides meaningful insights, and adheres to the word count i.e. 2400 to 2700 words. "
    "At the end of the essay, mention the various sources used in a brief citation format. "
    "The essay should be comprehensive, with clear organization and a professional tone."
    )

}

prompt_template1=("given a chat history and the latest user question"
            'which might reference context in the chat history,'
            'formulate a standalone question which can be understood'
            'without the chat history. Do not answer the question,'
            'just reformulate it if needed and otherwise return it as is.')

Prompt1=ChatPromptTemplate.from_messages([
    ('system',prompt_template1),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{input}')
])

model_selection=['Gemma2-9b-It','Llama-3.3-70b-Versatile','Gemma-7b-It','Llama3-70b-8192','Llama3-8b-8192','Mixtral-8x7b-32768','Llama-3.1-70b-Versatile','Llama-3.2-90b-Text-Preview']
with st.sidebar.form('Model Settings'):
    st.session_state.Temprature_Selection=st.slider('Randomness',min_value=0.01,max_value=1.0,step=0.05,value=0.7)
    st.session_state.selected_model=st.selectbox('Select Model',options=model_selection)
    st.form_submit_button('Submit')

List_Tab=list(prompts_dict.keys())
List_Tab=st.tabs(prompts_dict.keys())

llm_model=ChatGroq(model=st.session_state.selected_model,api_key=os.getenv('GROQ_API_KEY'),temperature=st.session_state.Temprature_Selection)
if 'store' not in st.session_state:
    st.session_state.store={}

prompt_template1=("given a chat history and the latest user question"
            'which might reference context in the chat history,'
            'formulate a standalone question which can be understood'
            'without the chat history. Do not answer the question,'
            'just reformulate it if needed and otherwise return it as is.')




def Session_History(session_id:str)->BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id]=ChatMessageHistory()
    return st.session_state.store[session_id]


def step1(session_id,temploc):
    documents=[]
    for uploaded_file in uploaded_files:
        temploc=temploc
        with open(temploc,'wb') as file:
            file.write(uploaded_file.getvalue())
            file_name=uploaded_file.name
            docs=PyPDFLoader(temploc).load()
            documents.extend(docs)
    splits=splitter.split_documents(documents=documents)
    Faiss_Vecorstore=FAISS.from_documents(documents=splits,embedding=HF_Embeddings)
    Faiss_Retriever=Faiss_Vecorstore.as_retriever()
    History_Aware_Ret=create_history_aware_retriever(llm=llm_model,retriever=Faiss_Retriever,prompt=Prompt1)
    Combine_Doc_Chain=create_stuff_documents_chain(llm=llm_model,prompt=Context_Prompt)
    Rag_Chain=create_retrieval_chain(retriever=History_Aware_Ret,combine_docs_chain=Combine_Doc_Chain)
    History=Session_History(session_id)
    my_agent=initialize_agent(
        tools=Tools,
        llm=llm_model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    Conv_Runnable_Chain=RunnableWithMessageHistory(
        runnable=Rag_Chain,
        get_session_history=Session_History,
        input_messages_key='input',
        output_messages_key='answer',
        history_messages_key='chat_history'
    )
    return Conv_Runnable_Chain, my_agent


def step2(session_id,temploc):
    documents=[]
    docs=PyPDFLoader(temploc).load()
    documents.extend(docs)
    splits=splitter.split_documents(documents=documents)
    Faiss_Vecorstore=FAISS.from_documents(documents=splits,embedding=HF_Embeddings)
    Faiss_Retriever=Faiss_Vecorstore.as_retriever()
    History_Aware_Ret=create_history_aware_retriever(llm=llm_model,retriever=Faiss_Retriever,prompt=Prompt1)
    Combine_Doc_Chain=create_stuff_documents_chain(llm=llm_model,prompt=Context_Prompt)
    Rag_Chain=create_retrieval_chain(retriever=History_Aware_Ret,combine_docs_chain=Combine_Doc_Chain)
    History=Session_History(session_id)
    my_agent=initialize_agent(
        tools=Tools,
        llm=llm_model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    Conv_Runnable_Chain=RunnableWithMessageHistory(
        runnable=Rag_Chain,
        get_session_history=Session_History,
        input_messages_key='input',
        output_messages_key='answer',
        history_messages_key='chat_history'
    )
    return Conv_Runnable_Chain, my_agent


Wikipedi_Tool=Tool(
    name='Wiki',
    func=WikipediaAPIWrapper().run,
    description='useful for when you need to answer questions about current events or the current state of the world'
    
) 

Arxiv_Tool=Tool(
    name='Arxiv',
    func=ArxivAPIWrapper().run,
    description='useful for when you need to answer questions about current events or the current state of the world'
)

Tools=[Wikipedi_Tool,Arxiv_Tool]

with List_Tab[0]:
    Context_Prompt=ChatPromptTemplate.from_messages([
        ('system',prompts_dict['PDF User Query']),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human','{input}')
    ])
    uploaded_files=st.file_uploader('Upload PDF',type=['pdf'],accept_multiple_files=True)
    if uploaded_files:
        user_query=st.text_input('User Query')
        temploc="./temp.pdf"
        if 'messages' not in st.session_state:
            st.session_state.messages=[]
        if user_query:
            st.chat_message('user').write(user_query)
        # st.session_state.messages.clear()
            with st.spinner("Generating response..."):
                st.session_state.messages.append({'role':'user','content':user_query})
                st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
                runnable_chain,my_agent=step1(session_id=session_id,temploc=temploc)
                response=runnable_chain.invoke(
                    {'input':user_query},config={'configurable':{'session_id':session_id}}
                )
                response_1=my_agent.run(user_query)      
                st.chat_message('').write(response['answer'])
                st.chat_message('ai').write(response_1)
                st.session_state.messages.append({'role':'assistant','content':response['answer']})
                st.session_state.messages.append({'role':'ai','content':response_1})
                st.image(image="chat.png")
                st.write('''🟢 ══════⊹⊱≼≽⊰⊹══════ ∘₊✧──────✧₊∘ ══════⊹⊱≼≽⊰⊹══════ 🟢''')
                for msg in st.session_state.messages:
                    st.chat_message(msg['role']).write(msg['content'])
                st.write('''🔴 ══════⊹⊱≼≽⊰⊹══════ ∘₊✧──────✧₊∘ ══════⊹⊱≼≽⊰⊹══════ 🔴''')


with List_Tab[1]:
    
    User_Query_Prompt=st.text_input('Find Prompt')
    Context_Prompt=ChatPromptTemplate.from_messages([
        ('system',prompts_dict['Prompt Generator']),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human','{input}')
    ])
    temploc="dummypdf.pdf"
    if 'messages' not in st.session_state:
        st.session_state.messages=[]
    if User_Query_Prompt:
        user_query=User_Query_Prompt
        st.chat_message('user').write(user_query)
        # st.session_state.messages.clear()
        with st.spinner("Generating response..."):
            st.session_state.messages.append({'role':'user','content':user_query})
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            runnable_chain,my_agent=step2(session_id=session_id,temploc=temploc)
            response=runnable_chain.invoke(
                {'input':user_query},config={'configurable':{'session_id':session_id}}
            )
            st.chat_message('').write(response['answer'])
            st.session_state.messages.append({'role':'assistant','content':response['answer']})
            st.image(image="chat.png")
            st.write('''🟢 ══════⊹⊱≼≽⊰⊹══════ ∘₊✧──────✧₊∘ ══════⊹⊱≼≽⊰⊹══════ 🟢''')
            for msg in st.session_state.messages:
                st.chat_message(msg['role']).write(msg['content'])
            st.write('''🔴 ══════⊹⊱≼≽⊰⊹══════ ∘₊✧──────✧₊∘ ══════⊹⊱≼≽⊰⊹══════ 🔴''')


with List_Tab[2]:
    base_template=prompts_dict['Movie Similarity Search']
    prompt=PromptTemplate(template=base_template,input_variables=['text'])
    
    my_agent=initialize_agent(
        tools=Tools,
        llm=llm_model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    chain=LLMChain(llm=llm_model,prompt=prompt)
    User_Query_Prompt=st.text_input('Provide Movie/Plot/Genre ?')
    if 'messages' not in st.session_state:
        st.session_state.messages=[]
    if User_Query_Prompt:
        input_data={'text':User_Query_Prompt}
        My_Response=chain.run(input_data)
        st.success(My_Response)

with List_Tab[3]:
    st.write('Under Development')





