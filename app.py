import streamlit as st
from streamlit_date_picker import date_range_picker, PickerType, Unit, date_picker
import numpy as np
import base64
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai


# ------------------------------------------- Config --------------------------------------------------------------
# Loading Local env variables
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
client = OpenAI(api_key=os.getenv("openai_api_key"))


# The code below is for the layout of the page
st.set_page_config(  # Alternate names: setup_page, page, layout
    # layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title='Travel Agent Chatbot',  # String or None. Strings get appended with "• Streamlit".
    page_icon=None,  # String, anything supported by st.image, or None.
)

# ------------------------------------------- Functions --------------------------------------------------------------

# Image from Local
path = os.path.dirname(__file__)
image_file = path+'/data/bg2.jpg'

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

def get_conversational_chain():

    prompt_template = """
    Assume you are a Professional Expert Travel Agent who helps the customers to plan their vacations/trips by suggesting the Flight Time and Flight Costs for given dates between the selected origin to destination places. 
    Please summarize the information like showing the flight time and costs as a json.\n\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.6)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def get_gemini_repsonse(input):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content(input, )
    return response.text


def fetch_travel_information_for_user_queries(query, travel_type):

    if travel_type == "General Travel Information":
      prompt_temp =  f"""prompt": "You are tasked with developing a travel planner system that handles multiple questions from the users like the distance between the cities, flight time, range of the cost of the flight. The system should act as a virtual travel planner, providing information about  distance between the cities, flight time and range of the cost of the flight when users inquire about travel options between places. The system should showcase virtual data, including  distance between the cities, flight time, range of the cost of the flight. The output response should be summarized and show information about the distance, flight time and range of the cost of flight.",
        "user_question": "{query}"""

    else:
      prompt_temp = f"""
            "prompt": "You are tasked with developing a travel planner system that handles multiple travel agent tour packages for various cities. The system should act as a virtual travel planner, providing information about tour packages when users inquire about travel options between places or for a specific destination. The system should showcase virtual data for each tour package, including cost, duration, what is included in the package, and if there are any family tour packages available. Additionally, include a category for 'package type' and output all the different types of packages offered by each agent.",
              "user_question" --> "{query}"
            """

    result = get_gemini_repsonse(prompt_temp)
    return result


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []


if "messages_gti" not in st.session_state:
    st.session_state.messages_gti = []

if "messages_tpi" not in st.session_state:
    st.session_state.messages_tpi = []


# ------------------------------------------- Setup --------------------------------------------------------------

# add_bg_from_local(image_file)

col1, col2, col3 = st.columns([0.5,2,0.5])
with col2:
    st.header("Travel Agent Chatbot :robot_face:")
    st.markdown("")
    st.markdown("")
c1, c2, c3 = st.columns([1,1,0.9])

with st.sidebar:
    st.title(":blue[Travel Agent Chatbot]")
    st.markdown("Your :orange[Personal Travel Assistant] designed to make planning your next trip a breeze! :red[Powered by LLM's], our chatbot harnesses the power of cutting-edge AI technology to provide information about the :orange[Distance, Flight Time, Flight Cost] between two cities, or the enticing :orange[Travel Package] options available for your next adventure.")
    st.markdown("")
    st.markdown("")
    travel_type = st.radio("What is your question about?", ["General Travel Information", "Tour Package Information"])

# ------------------------------------------- Output --------------------------------------------------------------

if travel_type == "General Travel Information":
    for message in st.session_state.messages_gti:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    for message in st.session_state.messages_tpi:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about your Travel Information? Ex: Flight Cost between Mumbai to Delhi"):
    # st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = fetch_travel_information_for_user_queries(prompt, travel_type)
        st.write(response)

    # st.session_state.messages.append({"role": "assistant", "content": response})

    if travel_type == "General Travel Information":
        st.session_state.messages_gti.append({"role": "user", "content": prompt})
        st.session_state.messages_gti.append({"role": "assistant", "content": response})
    else:
        st.session_state.messages_tpi.append({"role": "user", "content": prompt})
        st.session_state.messages_tpi.append({"role": "assistant", "content": response})