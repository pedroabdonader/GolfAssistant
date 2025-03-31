import os
import requests
import streamlit as st
from openai import AzureOpenAI

# Initialize your Azure OpenAI client
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
model_name = "gpt-4"
deployment = "gpt-4"

subscription_key = os.getenv("AZURE_OPENAI_KEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Azure AI Search Service configuration
search_service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
index_name = "honest-scooter-6jcrym1407"

# Initialize the conversation history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful RAG assistant. Only answer my questions based on the context I give you. Do not answer anything outside of the context. If the question can't be inferred from the context, please respond with 'Not available in existing content'"
        }
    ]

# Function to query Azure AI Search Service
def retrieve_documents(query):
    headers = {
        "Content-Type": "application/json",
        "api-key": search_api_key,
    }
    search_url = f"{search_service_endpoint}/indexes/{index_name}/docs?search={query}&api-version=2021-04-30-Preview"
    response = requests.get(search_url, headers=headers)
    if response.status_code == 200:
        return response.json().get("value", [])
    else:
        return []

# Function to get a response from the chatbot
def get_response(user_input):
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(user_input)
    context = "\n".join([doc['content'] for doc in retrieved_docs])  # Assuming 'content' holds the relevant text
    references = "\n".join([f"Source: {doc['title']} (ID: {doc['id']})" for doc in retrieved_docs])  # Assuming 'title' and 'id' are available

    # Append context to messages
    st.session_state.messages.append({"role": "system", "content": context})
    st.session_state.messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        stream=True,
        messages=st.session_state.messages,
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model=deployment,
    )

    # Collect the response
    assistant_response = ""
    for update in response:
        if update.choices:
            assistant_response += update.choices[0].delta.content or ""

    # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    # Add references to the response
    #if references:
    #    assistant_response += "\n\nReferences:\n" + references

    return assistant_response

# Streamlit UI
st.title("RAG Assistant")
st.write("Ask your questions below:")

# Create a text input with a key
user_input = st.text_input("You:", "", key="user_input")

# Use a button to send the input
if st.button("Send") or (user_input and st.session_state.get("user_input") != ""):
    response = get_response(user_input)
    st.write(f"**Assistant:** {response}")

client.close()