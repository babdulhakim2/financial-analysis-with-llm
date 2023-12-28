
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import json
import os
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
import uuid
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import re
import spacy
from keywords import keywords
nlp = spacy.load('en_core_web_lg')
# search = GoogleSearchAPIWrapper()
# google_tool = Tool(
#     name="Google Search",
#     description="Search Google for recent results.",
#     func=search.run
# )




def check_table_output_dict(output_dict):
    expected_keys = ['date', 'debit', 'credit', 'description']
    for key in expected_keys:
        if key not in output_dict or not output_dict[key]:
            return False
    return True



def get_table_columns(agent):
    """
    Query an agent and return the response as a dict.

    Args:
        agent: The agent to query.
        

    Returns:
        The response from the agent as a dictionary with values for `date`,`debit`,`credit`,`description`
    """


    transaction_template = """
    For the data, extract the following information:

    - Date: Date on which the transaction occured

    - Debit: The column name that represents outflow or debit from the bank account. If found, answer with the exact column name; otherwise, respond with "None" if not found or unknown.

    - Credit: The column name that represents inflow or credit to the bank account. If found, answer with the exact column name; otherwise, respond with "None" if not found or unknown.

    - Description: The column name that represents transaction details or description. Respond with the exact name in a case-sensitive manner.

    Format the output as JSON with the following keys:
      date
      debit
      credit
      description
      transaction_type
      

    data: {data}

    """

    date = ResponseSchema(name="date",
                             description="The column name that represents \
                             the date on which the transaction occured.")

    debit = ResponseSchema(name="debit",
                             description="The column name that represents \
                             outflow or debit from the bank account. If found, \
                                answer with the exact column name; otherwise,\
                                  respond with `None` if not found or unknown.")
    credit = ResponseSchema(name="credit",
                                          description="The column name that represents \
                                          inflow or credit to the bank account. If found, \
                                          answer with the exact column name; otherwise, \
                                          respond with `None` if not found or unknown.")
    description = ResponseSchema(name="description",
                                    description="The column name that represents \
                                    transaction details or description, may include name of entity transacted with. \
                                    Respond withthe exact name in a case-sensitive manner.")
    transaction_type = ResponseSchema(name="transaction_type",
                                    description="The column name that represents \
                                    transaction type. Return None if not found \
                                    Respond withthe exact name in a case-sensitive manner.")


    response_schemas = [date,
                        debit, 
                        credit,
                        description,
                        transaction_type]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()


    prompt = ChatPromptTemplate.from_template(template=transaction_template)

    messages = prompt.format_messages(data=agent, 
                                    format_instructions=format_instructions)
    chat = ChatOpenAI(temperature=0.0)
    response = chat(messages)
    
    output_dict = output_parser.parse(response.content)


    if not check_table_output_dict(output_dict):
        return False



  
    return output_dict

def get_transaction_summary(df, table_columns, top_ten_transactions, account_owner=""):
    
    prompt = """
    Please generate a detailed summary of the account statement for the given data`, \
        Please format the currency amounts with comma separation and use ₦ as currency  
        and the dates in the format like "6th January, 2023". For the top_ten_transactions, \
            observe any trend associated with the account. data: {data}
    """
    prompt_template = ChatPromptTemplate.from_template(prompt)


    total_credit = df[table_columns['credit']].sum()
    total_debit = df[table_columns['debit']].sum()
    

    # Get the date range of transaction activity
    date_range = f"{df[table_columns['date']].min()} to {df[table_columns['date']].max()}"

    # Get the total number of transactions
    transaction_count = len(df)
    data = {"account_owner":account_owner,"total_credit": total_credit, "total_debit": total_debit, "date_range": date_range, "transaction_count": transaction_count, "top_ten_transactions": top_ten_transactions}

    messages = prompt_template.format_messages(data=data)
    chat = ChatOpenAI(temperature=0.0)
    response = chat(messages)
    return response.content
  



def get_suspicious_trends_summary(df, table_columns, top_ten_transactions, account_owner=""):
    
    prompt = """
    Please generate a detailed summary of the account statement for the given data,\
          with a focus on identifying any suspicious activities. The currency amounts\
              should be formatted with comma separation and use ₦ as the currency symbol.
                Please represent dates in the format "6th January, 2023". 
                Analyze the following bank statement for any potential suspicious trends:

{data}
    """
    prompt_template = ChatPromptTemplate.from_template(prompt)


    total_credit = df[table_columns['credit']].sum()
    total_debit = df[table_columns['debit']].sum()
    

    # Get the date range of transaction activity
    date_range = f"{df[table_columns['date']].min()} to {df[table_columns['date']].max()}"

    # Get the total number of transactions
    transaction_count = len(df)
    data = {"account_owner":account_owner,"total_credit": total_credit, "total_debit": total_debit, "date_range": date_range, "transaction_count": transaction_count, "top_ten_transactions": top_ten_transactions}

    messages = prompt_template.format_messages(data=data)
    chat = ChatOpenAI(temperature=0.0)
    response = chat(messages)
    return response.content
  


def get_mltf_hypothesis(df, table_columns, top_ten_transactions, account_owner=""):
    
    prompt = """
    Please generate a seeveral detailed hypothesis on the given account statement data in reelation to money laundering or terrorism financing`. \
        All amounts should be presented with comma separation and ₦ as currency,  
        and any the date should be in the format like "6th January, 2023". data: {data}
    """
    prompt_template = ChatPromptTemplate.from_template(prompt)


    total_credit = df[table_columns['credit']].sum()
    total_debit = df[table_columns['debit']].sum()
    

    # Get the date range of transaction activity
    date_range = f"{df[table_columns['date']].min()} to {df[table_columns['date']].max()}"

    # Get the total number of transactions
    transaction_count = len(df)
    data = {"account_owner":account_owner,"total_credit": total_credit, "total_debit": total_debit, "date_range": date_range, "transaction_count": transaction_count, "top_ten_transactions": top_ten_transactions}

    messages = prompt_template.format_messages(data=data)
    chat = ChatOpenAI(temperature=0.0)
    response = chat(messages)
    return response.content
  

def get_top_ten_transactions(df, table_columns):

    # Group the data by transaction description and calculate transaction count, total amount, and date range
    grouped_data = df.groupby(table_columns['description']).agg({table_columns['debit']: 'sum', table_columns['credit']: 'sum', table_columns['date']: ['count', 'min', 'max']})

    # Rename the columns for clarity
    grouped_data.columns = [table_columns['debit'], table_columns['credit'], 'Transaction Count', 'Date Min', 'Date Max']

    # Sort the grouped data by transaction count in descending order
    sorted_data = grouped_data.sort_values('Transaction Count', ascending=False)

    # Get the top 10 unique transactions
    top_10_transactions = sorted_data.head(10)

    # Add a column for total amount
    top_10_transactions['Total Amount'] = top_10_transactions['Debit'] + top_10_transactions['Credit']

    return top_10_transactions
    


def find_cash_and_deposits(df, description, transaction_type):
    # Define keywords and pattern for cash deposits
    cash_keywords = ['cash', 'deposit']

    # Regular expression pattern for keyword matching
    cash_pattern = re.compile(r'\b(?:{})\b'.format('|'.join(cash_keywords)), flags=re.IGNORECASE)

    # Extract cash deposit transactions based on keyword matching in description column
    cash_deposits_desc = df[df[description].str.contains(cash_pattern, na=False)]

    if transaction_type != str(None):
        # Extract cash deposit transactions based on keyword matching in transaction_type column
        cash_deposits_type = df[df[transaction_type].str.contains(cash_pattern, na=False)]

        # Combine the results from both columns
        cash_deposits = pd.concat([cash_deposits_desc, cash_deposits_type])
        return cash_deposits
    else:
        return cash_deposits_desc



def process_entities_and_search(df, description):
    nlp = spacy.load("en_core_web_lg")
    doc_entities = []

    for index, row in df.iterrows():
        doc = nlp(row[description])

        entities = []
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))

        doc_entities.append(entities)

    return doc_entities


def display_entities(entities):
    persons = set()
    organizations = set()
    locations = set()
    miscellaneous = set()

    for doc_entities in entities:
        for entity, label in doc_entities:
            if label == 'PERSON':
                persons.add(entity)
            elif label == 'ORG':
                organizations.add(entity)
            elif label == 'LOC':
                locations.add(entity)
            else:
                miscellaneous.add(entity)
    

    if persons:
        st.subheader('Persons')
        st.write(', '.join(persons))

    if organizations:
        st.subheader('Organizations')
        st.write(', '.join(organizations))

    if locations:
        st.subheader('Locations')
        st.write(', '.join(locations))

    if miscellaneous:
        st.subheader('Miscellaneous')
        st.write(', '.join(miscellaneous))


def check_adverse_dict(output_dict):
    expected_keys = ['adverse', 'reason']
    for key in expected_keys:
        if key not in output_dict or not output_dict[key]:
            return False
    return True



def analyze_content(content):
    # Invoke the OpenAI Language Model to analyze the content and determine if it's adverse
    adverse_template = """
    For the given content, give me the following information:

    - adverse: True if the content is adverse, False otherwise

    - reason: Reason for flaging content as adverse

    Format the output as JSON with the following keys:
      adverse
      reason
      

    content: {content}

    """
    adverse = ResponseSchema(name="adverse",
                             description="Wether the content contain adverse.")

    reason = ResponseSchema(name="reason",
                             description="Why it is considered adverse.")
    


    response_schemas = [adverse,
                        reason]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()


    prompt = ChatPromptTemplate.from_template(template=adverse_template)

    messages = prompt.format_messages(content=content, 
                                    format_instructions=format_instructions)
    chat = ChatOpenAI(temperature=0.0)
    response = chat(messages)
    
    output_dict = output_parser.parse(response.content)


    if not check_adverse_dict(output_dict):
        return False

    return output_dict



# def detect_adverse_content(name):
#     adverse_results = []
#     query_results = []
#     for keyword in keywords:
#         query = name + " " + keyword
#         query_results.extend(google_tool.run(query))
    
#     for result in query_results:
#         result_label = analyze_content(result)
#         if result_label['adverse']:
#             adverse_results.append(result)
#     return adverse_results




def main():
    load_dotenv()
    # search = GoogleSearchAPIWrapper()


    st.set_page_config(page_title="Scolo", page_icon=":moneybag:")
    st.markdown("""
        <style>
        .subtitle {
            font-size: 20px;
            color: #555555;
        }
        </style>
        """, unsafe_allow_html=True)
    st.header("🦜🔗 AI Intelligence Analyst")
    st.markdown('<p class="subtitle">Upload a flat file with headers as the first line for optimal results. Rest assured, your documents are read locally on your device, ensuring strict privacy compliance.</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an Excel file", type='xlsx')

    if "rerun_counter" not in st.session_state:
        st.session_state.rerun_counter = 0



    if uploaded_file is not None:
        unique_filename = str(uuid.uuid4()) + '.xlsx'
        uploads_folder = 'uploads'
        os.makedirs(uploads_folder, exist_ok=True)  # Create 'uploads' folder if it doesn't exist
        file_path = os.path.join(uploads_folder, unique_filename)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        df = pd.read_excel(uploaded_file)
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
        if st.checkbox("Show raw data"):
            with st.expander('Data'):
                st.table(df.style.set_properties(**{'text-align': 'center'}).set_table_styles([{
                    'selector': 'th',
                    'props': [('text-align', 'center')]
                }]))

        account_owner = st.text_input("Enter subject's name to help the AI understand the context of the data (e.g. 'John Doe') ")

        if not get_table_columns(agent):
            st.error("Unable to extract table columns. Please ensure that the file has columns for `date`, `debit`, `credit`, and `description`.")
            st.stop()
        if account_owner:
            with st.spinner(text="Analysing document..."):
                table_columns = get_table_columns(agent)
                top_ten_transactions = get_top_ten_transactions(df, table_columns)
                with st.expander('Account summary'):
                    with st.spinner(text="Analysing..."):  
                        transaction_summary = get_transaction_summary(df, table_columns, top_ten_transactions, account_owner=account_owner)
                        st.info(transaction_summary)
                        st.success("Done!")

                with st.expander('Suspicious trends'):
                    with st.spinner(text="Detecting suspicious activity trends..."):  
                        suspicious_trends_summary = get_suspicious_trends_summary(df, table_columns, top_ten_transactions, account_owner=account_owner)
                        st.info(suspicious_trends_summary)
                        st.success("Done!")

                with st.expander('Cash and Deposits'):
                    with st.spinner(text="Detecting cash and deposits..."):  
                        cash_deposits = find_cash_and_deposits(df, table_columns['description'], table_columns['transaction_type'])
                        if not cash_deposits.empty:
                            cash_deposits = cash_deposits.reset_index(drop=True)
                            st.dataframe(cash_deposits.style.format({'Amount': '{:,.2f}'}), height=400)
                        else:
                            st.info("No cash deposits found.")
                        st.success("Done!")


             
                with st.expander('Entities'):
                    with st.spinner(text="Detecting entities..."):
                        entities = process_entities_and_search(df, table_columns['description'])
                        display_entities(entities)
                        st.success("Done!")


                with st.expander('ML/TF hypothesis on account statement'):
                    with st.spinner(text="Detecting ML/TF..."):
                        mltf_hypothesis = get_mltf_hypothesis(df, table_columns, top_ten_transactions, account_owner=account_owner)
                        st.info(mltf_hypothesis)
                        st.success("Done!")











        
        

if __name__ == "__main__":
    main()