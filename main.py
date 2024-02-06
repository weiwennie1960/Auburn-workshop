import pandas as pd
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

import os
import re

import streamlit as st
from dotenv import load_dotenv

# Set up OpenAI API key
load_dotenv()
API_KEY = os.getenv("API_KEY")


# Set up Streamlit

st.header("Onet KSAs Alignment Tool\n Powered by GPT Language model\n\n\n")

st.write("#### Job Description Input")

# Define a function to break down job description into a list

def split_to_sentence(text):
    if text.endswith("\n"):
      text = text[:-1] + "."
    sentence_list = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?)\s', text)
    return (sentence_list)

#_______________________________________________________________________________________
#load prompt template to memery
with open("Onet KSAs.txt") as f:
    template = f.read()
    f.close()

#create a new chain to line up all the elements

prompt = ChatPromptTemplate.from_template(template
)
output_parser = StrOutputParser()
model = ChatOpenAI(model="gpt-4-1106-preview", api_key=API_KEY)
chain = (
    {"jd": RunnablePassthrough()} 
    | prompt
    | model
    | output_parser
)
#_______________________________________________________________________________________

# Define a function to input job description

with st.form(key='jd_form'):
    input_text_jd = st.text_area(label="Job Description Input", placeholder="Job Description....", key="jd_input")
    submit_button = st.form_submit_button(label='Submit')


if submit_button:
    jd_sentence = split_to_sentence(input_text_jd)

    #st.write(input_text_jd)
    #use a for loop to put all the jd into the chain and then store them into a list
    ksa_list = []
    total_num = len(jd_sentence)
    #create a progress bar
    progress_text = "Processing Job Description. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for sentence in jd_sentence:
        ksa_list.append(chain.invoke(sentence))
        #update the total number of jd left
        total_num -= 1
        percent_complete = (len(jd_sentence) - total_num) / len(jd_sentence)
        progress_text = f"Processing Job Description. {total_num} job description left."
        my_bar.progress(percent_complete, text=progress_text)

    my_bar.empty()

    #create a dataframe put jd_sentence and ksa_list into columns
    df = pd.DataFrame({"Job Description":jd_sentence,"KSAs":ksa_list})
    st.dataframe(df)

    #output unique element from the ksa_list
    unique_ksa = set(ksa_list)
    # count each element in the ksa_list
    ksa_count = []
    for i in unique_ksa:
        ksa_count.append(ksa_list.count(i))
    #output time each element appears in the ksa_list
    st.write("Unique KSAs:")
    for i in unique_ksa:
        st.write(i + ": " + str(ksa_list.count(i)) + " time(s)")