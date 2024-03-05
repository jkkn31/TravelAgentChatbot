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

# Loading Local env variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# The code below is for the layout of the page
st.set_page_config(  # Alternate names: setup_page, page, layout
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title='Travel Agent Chatbot',  # String or None. Strings get appended with "• Streamlit".
    page_icon=None,  # String, anything supported by st.image, or None.
)

st.header("Hello")

# st.image("data/bg.jpg")

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

add_bg_from_local(image_file)

with st.sidebar:

    st.title("Travel Agent Chatbot!")
    ttype = st.radio("What are you looking for?", ("Flights", "Vacation Plan"))
    # z = st.selectbox("from", ["Hello", "how"])
    if ttype == "Flights":
        cc1, cc2 = st.columns(2)
        places = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Boston", "Maldives", "Paris", "Bali", "Santorini", "Tokyo", "Rome", "New York", "Dubai", "UAE", "Barcelona", "Sydney"]
        with cc1:
            originp = st.selectbox("Origin", places)
        with cc2:
            # dplace = places.remove(originp)
            destinationp = st.selectbox("Destination", places)

        c1, c2, c3 = st.columns(3)

        # with c1:
            # Use date_picker to create a date picker
        date_string = date_picker(picker_type=PickerType.time.string_value, value=0, unit=Unit.days.string_value,
                                  key='date_picker')

        if date_string is not None:
            st.write('Date Picker: ', date_string)

            # Use date_range_picker to create a datetime range picker
        st.subheader('Date Range Picker')
        date_range_string = date_range_picker(picker_type=PickerType.time.string_value,
                                              start=-30, end=0, unit=Unit.minutes.string_value,
                                              key='range_picker',
                                              refresh_button={'is_show': True, 'button_name': 'Refresh last 30min',
                                                              'refresh_date': -30,
                                                              'unit': Unit.minutes.string_value})
        if date_range_string is not None:
            start_datetime = date_range_string[0]
            end_datetime = date_range_string[1]
            st.write(f"Date Range Picker [{start_datetime}, {end_datetime}]")

            # month = st.selectbox("Month", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        with c2:
            month = st.selectbox("Day", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
        with c3:
            year = st.selectbox("Year", [2024, 2025])

    # print(z)

# st.container()

# st.chat("Hello")

col1, col2, col3 = st.columns([0.5, 6, 0.5])
Z = st.container()
with Z:
    # with st.chat_message("User"):
    #     st.write("Hello human")
    #     st.bar_chart(np.random.randn(30, 3))
    #
    # prompt = st.chat_input("Say something")
    # if prompt:
    #     st.write(f"User has sent the following prompt: {prompt}")


    st.title("ChatGPT-like clone")

    client = OpenAI(api_key=os.getenv("openai_api_key"))

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})