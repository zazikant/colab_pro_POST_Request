from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import openai
import pprint
import json
import pandas as pd
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import re

import requests
import csv

import matplotlib.pyplot as plt
import io

load_dotenv(find_dotenv())

load_dotenv()

from dotenv import find_dotenv, load_dotenv

import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

from langchain.document_loaders import PyPDFLoader

from dotenv import load_dotenv

import os
import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.memory import ConversationSummaryBufferMemory

from langchain.chains.summarize import load_summarize_chain


# Laden Sie die Umgebungsvariablen aus der .env-Datei
load_dotenv()
API_KEY = os.environ.get("API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
openai.api_key = os.environ["OPENAI_API_KEY"]

def draft_email(user_input):

    from langchain.document_loaders import DirectoryLoader, CSVLoader

    loader = DirectoryLoader(
        "./shashi", glob="**/*.csv", loader_cls=CSVLoader, show_progress=True
    )
    docs = loader.load()

    #textsplitter-----------------

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=2,
    )

    docs = text_splitter.split_documents(docs)
    # print(docs[3].page_content)
    #-----------------

    from langchain.embeddings import OpenAIEmbeddings
    openai_embeddings = OpenAIEmbeddings()

    from langchain.vectorstores.faiss import FAISS
    import pickle

    #Very important - db below is used for similarity search and not been used by agents in tools

    db = FAISS.from_documents(docs, openai_embeddings)
    
    import pickle

    with open("db.pkl", "wb") as f:
        pickle.dump(db, f)
        
    with open("db.pkl", "rb") as f:
        db = pickle.load(f)
        
    query = user_input
    docs = db.similarity_search(query, k=8)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


    # template = """
    # you are a pediatric dentist and you are writing a key features serial wise for following information: 

    # text: {context}
    # """    
    map_prompt = """
    Write a concise summary of the following:
    "{text}"
    CONCISE SUMMARY:
    """
    
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    
    combine_prompt = """
    You are a summarisation expert. Focus on maintaining a coherent flow and using proper grammar and language. Write a detailed summary of the following text:
    "{text}"
    SUMMARY:
    """
    
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    
    summary_chain = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template, verbose=True
                                    )

    response = summary_chain.run({"input_documents": docs})

    return response


def extract_email(user_input):
    # Regular expression pattern to match email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # Search for email pattern in the user input
    match = re.search(email_pattern, user_input)
    
    if match:
        email = match.group(0)  # Extract the email address
        return email
    
    return None