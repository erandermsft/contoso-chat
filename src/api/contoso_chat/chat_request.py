from dotenv import load_dotenv
load_dotenv()

from azure.identity import DefaultAzureCredential

from azure.cosmos import CosmosClient
import os
from contoso_chat.product import product
import prompty.azure
from prompty.tracer import trace

from langchain_openai import AzureChatOpenAI
from pathlib import Path
from langchain_prompty import create_chat_prompt
from langchain_core.output_parsers import StrOutputParser

def azure_ad_token_provider():
    # Initialize Azure Credential
    credential = DefaultAzureCredential()
    
    # Obtain the token with the correct scope
    token = credential.get_token("https://cognitiveservices.azure.com/.default").token
    
    return token

def get_customer(customerId: str) -> str:
    
    try:
        url = os.environ["COSMOS_ENDPOINT"]
        client = CosmosClient(url=url, credential=DefaultAzureCredential())
        db = client.get_database_client("contoso-outdoor")
        container = db.get_container_client("customers")
        response = container.read_item(item=str(customerId), partition_key=str(customerId))
        response["orders"] = response["orders"][:2]
        return response
    except Exception as e:
        print(f"Error retrieving customer: {e}")
        return None

"""
@trace
def get_customer(input_dict: str) -> str:
    
    # Get customer id
    customerId = input_dict.get("customerId", "")

    try:
        url = os.environ["COSMOS_ENDPOINT"]
        client = CosmosClient(url=url, credential=DefaultAzureCredential())
        db = client.get_database_client("contoso-outdoor")
        container = db.get_container_client("customers")
        response = container.read_item(item=str(customerId), partition_key=str(customerId))
        response["orders"] = response["orders"][:2]
        return response
    except Exception as e:
        print(f"Error retrieving customer: {e}")
        return None

def get_question(input_dict: str) -> str:
    return input_dict.get("question", "")

def get_chat_history(input_dict: str) -> str:
    return input_dict.get("chat_history", [])


# Defining Langchain components
def get_respone_chain():
    # Load prompty as langchain ChatPromptTemplate
    folder = Path(__file__).parent.absolute().as_posix()
    path_to_prompty = folder + "/chat.prompty"
    prompt = create_chat_prompt(path_to_prompty)

    # Initialize the Azure OpenAI model
    model = AzureChatOpenAI(
        deployment_name="gpt-4",
        api_version="2023-06-01-preview",
        api_key=azure_ad_token_provider(),
        openai_api_type="azure_ad"
    )

    output_parser = StrOutputParser()

    get_response = {"customer": get_customer, "documentation": product.find_products, "question":get_question, "chat_history": get_chat_history} | prompt | model | output_parser

    return get_response
"""

@trace
def get_response(customerId, question, chat_history):
    customer = get_customer(customerId)
    context = product.find_products(question)

    # Load prompty as langchain ChatPromptTemplate
    folder = Path(__file__).parent.absolute().as_posix()
    path_to_prompty = folder + "/chat.prompty"
    prompt = create_chat_prompt(path_to_prompty)

    # Initialize the Azure OpenAI model
    model = AzureChatOpenAI(
        deployment_name="gpt-4",
        api_version="2023-06-01-preview",
        api_key=azure_ad_token_provider(),
        openai_api_type="azure_ad"
    )

    output_parser = StrOutputParser()

    chain = prompt | model | output_parser
    
    result = chain.invoke({
        "customer": customer,
        "question": question,
        "chat_history": chat_history
    })

    return {"question": question, "answer": result, "context": context}
