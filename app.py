import streamlit as st
import base64
import os
import google.generativeai as genai
from dotenv import load_dotenv
import google.generativeai as genai
import sqlite3
import pandas as pd

# ------------------------------------------- Config --------------------------------------------------------------
# Loading Local env variables
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# The code below is for the layout of the page
st.set_page_config(  # Alternate names: setup_page, page, layout
    layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    page_title='Travel Agent Chatbot',  # String or None. Strings get appended with "• Streamlit".
    page_icon=None,  # String, anything supported by st.image, or None.
)

# ------------------------------------------- Functions --------------------------------------------------------------

# Image from Local
path = os.path.dirname(__file__)
image_file = path+'/data/bg2.jpg'


tour_package_df_cols = ['id', 'AGENT_NAME', "PACKAGE_TYPE", "INCLUSION", "COST", "DURATION"]
intent = ""

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


def get_gemini_repsonse(input):
    model=genai.GenerativeModel('gemini-pro')
    response=model.generate_content(input, )
    return response.text

def read_sql_query(sql,db):
    conn=sqlite3.connect(db)
    cur=conn.cursor()
    cur.execute(sql)
    rows=cur.fetchall()
    conn.commit()
    conn.close()
    return rows

def fetch_travel_information_for_user_queries(query, travel_type):

    if travel_type == "General Travel Information":
      prompt_temp =  f"""prompt": "You are tasked with developing a travel planner system that handles multiple questions from the users like the distance
       between the cities, flight time, range of the cost of the flight. The system should act as a virtual travel planner, providing information about 
        distance between the cities, flight time and range of the cost of the flight when users inquire about travel options between places. The system 
        should showcase virtual data, including  distance between the cities, flight time, range of the cost of the flight. The output response 
        should be summarized and show information about the distance, flight time and range of the cost of flight.",
        "user_question": "{query}"""

    elif travel_type=="Tour Package Information":
      prompt_temp = f"""
            "prompt": "You are tasked with developing a travel planner system that handles multiple travel agent tour packages for various cities. 
            The system should act as a virtual travel planner, providing information about tour packages when users inquire about travel options between 
            places or for a specific destination. The system should showcase virtual data for each tour package, including cost, duration, what is 
            included in the package, and if there are any family tour packages available. Additionally, include a category for 'package type' and 
            output all the different types of packages offered by each agent.",
              "user_question" --> "{query}"
            """
    else:
        prompt_temp = f"""
        "prompt": "You are tasked with developing a Intent Detector for helping the user to plan their travel. 
            The system should act as a virtual Intent Detector travel planner and if the user query is about including distance between the cities, flight time, range of the cost of the flight then the response is "General Travel Information". 
            If the user query is about vacation travel or vacation plan or tour plan options between places or for a specific destination then the response should be "Tour Package Information. The output should be presented as ##Intent: Response in this format where Response is your output.",
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

# Fetch General Travel Information
def fetch_general_travel_information_for_user_queries(query):
    prompt_temp = f"""prompt": "You are tasked with developing a travel planner system that handles multiple questions from the users like the distance
           between the cities, flight time, range of the cost of the flight. The system should act as a virtual travel planner, providing information about 
            distance between the cities, flight time and range of the cost of the flight when users inquire about travel options between places. The system 
            should showcase virtual data, including  distance between the cities, flight time, range of the cost of the flight. The output response 
            should be summarized and show information about the distance, flight time and range of the cost of flight for every user query.",
            "user_question": "{query}"""

    result = get_gemini_repsonse(prompt_temp)
    return result

# Fetch the Package related info from the db by using the query --> SQL Code --> Fetch from db
def fetch_vacation_plan_info_query(query):
    sql_prompt = f"""
        You are an expert in converting English questions to SQL query!
        The SQL database has the table name PACKAGE and has the following schema - (id int, AGENT_NAME VARCHAR(25),PACKAGE_TYPE VARCHAR(30),
    INCLUSION VARCHAR(120),COST INT, DURATION INT) and all the values in the table are in lowercase.\n\nFor example,\nExample 1 - How many entries of records are present?, 
        the SQL command will be something like this SELECT COUNT(*) FROM PACKAGE ;
        \nExample 2 - Show me all the tour packages for 5 days?, 
        the SQL command will be something like this SELECT * FROM PACKAGE WHERE DURATION = 5; 
        also the sql code should not have ``` in beginning or end and sql word in output and the output should be only the excuatable SQL code,
        "user_question" --> "{query}"
        """
    sql_code = get_gemini_repsonse(sql_prompt)

    sql_response = read_sql_query(sql_code, "tourpackage.db")
    # if sql_response > 0:
    # for row in sql_response:
    #     st.write(row)

    df = pd.DataFrame(sql_response, columns = tour_package_df_cols)

    # st.dataframe(df)
    print(df)
    return df


# To find the Intent of the user by the query! Step-1
def find_user_intent(query):
    prompt_temp = f"""
            "prompt": "You are tasked with developing a Intent Detector for helping the user to plan their travel. 
                The system should act as a virtual Intent Detector travel planner and if the user query is about including distance between the cities, flight time, range of the cost of the flight then the response is "General Travel Information". 
                If the user query is about vacation travel or vacation plan or tour plan options between places or for a specific destination then the response should be "Tour Package Information. The output should be presented as ##Intent: Response in this format where Response is your output.",
                "user_question" --> "{query}"
            """
    result = get_gemini_repsonse(prompt_temp)
    return result


# ------------------------------------------- Setup --------------------------------------------------------------

# add_bg_from_local(image_file)

col1, col2, col3 = st.columns([2.3,2,2])
with col2:
    st.header("Travel Agent Chatbot :robot_face:")
    st.markdown("")
st.markdown("Your :orange[Personal Travel Assistant] designed to make planning your next trip a breeze! :red[Powered by LLM's], our chatbot harnesses the power of cutting-edge AI technology to provide information about the :orange[Distance, Flight Time, Flight Cost] between two cities, or the enticing :orange[Travel Package] options available for your next adventure.")
st.markdown("")
c1, c2, c3 = st.columns([1,1,0.9])

# ------------------------------------------- Output --------------------------------------------------------------

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if travel_type1 == "General Travel Information":
#     for message in st.session_state.messages_gti:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
# else:
#     for message in st.session_state.messages_tpi:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])


def populate_the_results(intent):
    if intent == "General Travel Information":
        for message in st.session_state.messages_gti:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    elif intent == "Tour Package Information":
        for message in st.session_state.messages_tpi:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        pass

if prompt := st.chat_input("Ask me anything about your Travel Information? Ex: Flight Cost between Mumbai to Delhi"):
    # st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating the response..."):
            intent_response = find_user_intent(prompt) #1. Intent Detection using LLM
            intent = intent_response.split(": ")[-1]
            st.write(f"The Intent of the User Query is: :red[{intent}]")
            if intent == 'Tour Package Information':
                response = fetch_vacation_plan_info_query(prompt)
                st.dataframe(response)
            elif intent == "General Travel Information":
                response = fetch_general_travel_information_for_user_queries(prompt)
                st.write(response)
            else:
                st.write("User Query is not Found")


    # st.session_state.messages.append({"role": "user", "content": prompt})
    # st.session_state.messages.append({"role": "assistant", "content": response})
    #
    # if intent == "General Travel Information":
    #     st.session_state.messages_gti.append({"role": "user", "content": prompt})
    #     st.session_state.messages_gti.append({"role": "assistant", "content": response})
    # else:
    #     st.session_state.messages_tpi.append({"role": "user", "content": prompt})
    #     st.session_state.messages_tpi.append({"role": "assistant", "content": response})

