import streamlit as st
from streamlit_chat import message
import tempfile   # temporary file
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain_community.document_loaders.csv_loader import CSVLoader  # using CSV loaders
from langchain_community.embeddings import HuggingFaceEmbeddings # import hf embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
import plotly.figure_factory as ff
import plotly.express as px

DB_FAISS_PATH = 'vectorstore/db_faiss' # # Set the path of our generated embeddings


# Loading the model of your choice
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        config=
        {
            'max_new_tokens': 1024,
            'temperature': 0.1,
            'context_length': 2024
        }
                      
    )
    # the model defined, can be replaced with any ... vicuna,alpaca etc
    # name of model
    # tokens
    # the creativity parameter
    return llm


st.title("Zobo here!!")
# st.subheader("Hey i'm zobo powered by llama 2 model and with power of streamlit GUI XD, have fun")
st.markdown("<p style='text-align: left; color: white;'>ey i'm zobo powered by llama 2 model and with power of streamlit GUI XD, have fun</p>",unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload File", type="csv") # uploaded file is stored here
# file uploader
if uploaded_file:
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.getvalue())
        selected_file = tfile.name
        df = pd.read_csv(uploaded_file)

    loader = CSVLoader(file_path=selected_file, encoding="utf-8", csv_args={'delimiter': ','}) 
    
    data = loader.load() 
    #st.json(data)   

    # Split the text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cpu'}) # use sentence transformer to create embeddings

    # FAISS Can be replaced by Chroma... so it will be like CHROMA.fromdocuments...
    db = FAISS.from_documents(text_chunks, embeddings) # pass data embeddings vector data here
    db.save_local(DB_FAISS_PATH) # save vector embedding here on mentioned path
    llm = load_llm() # Load the Language model here

    # the conversational chain which preserves context learning in chat
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
    # ConversationalRetrievalChain can be replaced by LLMChain,retrivalQA
  

    # # Perform statistical analysis
    # mean_val = df.mean()
    # median_val = df.median()
    # mode_val = df.mode()
    # std_val = df.std()
    # corr_val = df.corr()

    # # Create a string representation of the results
    # result_str = ""
    # result_str += "Mean: {}\n".format(mean_val)
    # result_str += "Median: {}\n".format(median_val)
    # result_str += "Mode: {}\n".format(mode_val)
    # result_str += "Standard Deviation: {}\n".format(std_val)
    # result_str += "Correlation Coefficient: {}\n".format(corr_val)

    # # Append the result to the history list
    # query = "Statistical analysis of uploaded data"
    # result = {"answer": result_str}
    # if 'history' not in st.session_state:
    #     st.session_state['history'] = []
    # st.session_state['history'].append((query, result))



    # Perform basic statistical analysis show in web ui
    st.header("Statistical Analysis")
    st.markdown("<p style='text-align: left; color: white;'>Here we will show you some statistical analysis related to your CSV for reference, also our model can remember your last conversation.</p>",unsafe_allow_html=True)

    st.write("Mean:")
    st.write(df.mean())
    st.write("Median:")
    st.write(df.median())
    st.write("Mode:")
    st.write(df.mode())
    st.write("Standard Deviation:")
    st.write(df.std())
    st.write("Correlation Coefficient:")
    st.write(df.corr())

    

    # Generate plots
    st.header("Plots")
    plot_type = st.selectbox("Select a plot type", ["Histogram", "Scatter Plot", "Line Plot","Bar Plot"])
    if plot_type == "Histogram":
        value1 = st.selectbox("Select x-axis column", df.columns)
        value2 = st.selectbox("Select y-axis column", df.columns)
        value3 = st.selectbox("Select xz-axis column", df.columns)
        # fig, ax = plt.subplots()
        # ax.hist(df[value], bins=50)  
        # st.pyplot(fig)
        hist_data = [df[value1], df[value2], df[value3]]
        group_labels = [value1,value2,value3]
        # plot_str = "Histogram of {}".format(value)
        fig = ff.create_distplot(
        hist_data, group_labels, bin_size=[.1, .25, .5])
        st.plotly_chart(fig, use_container_width=True)
        # query = "Generate plot"
        # result = {"answer": plot_str, "plot": fig}
        # if 'history' not in st.session_state:
        #     st.session_state['history'] = []
        # st.session_state['history'].append((query, str(result)))
    elif plot_type == "Scatter Plot":
        x_axis = st.selectbox("Select x-axis column", df.columns)
        y_axis = st.selectbox("Select y-axis column", df.columns)
        fig, ax = plt.subplots()
        ax.scatter(df[x_axis], df[y_axis])
        st.pyplot(fig)
        plot_str = "Scatter plot of {} vs {}".format(x_axis, y_axis)
        query = "Generate plot"
        result = {"answer": plot_str, "plot": fig}
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append((query, str(result)))
    elif plot_type == "Line Plot":
        x_axis = st.selectbox("Select a column", df.columns)
        y_axis = st.selectbox("Select b column", df.columns)
        # fig, ax = plt.subplots()
        # ax.plot(df[x_axis], df[y_axis])
        chart_data = pd.DataFrame({
            f"{x_axis}": df[x_axis],
            f"{y_axis}": df[y_axis]
        })
        # st.pyplot(fig)
        st.line_chart(chart_data)
        plot_str = "Line plot of {} vs {}".format(x_axis, y_axis)
        query = "Generate plot"
        result = {"answer": plot_str, "plot": plot_type}
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        st.session_state['history'].append((query, str(result)))


    elif plot_type == "Bar Plot":
        value1 = st.selectbox("Select column 1", df.columns)
        value2 = st.selectbox("Select column 2", df.columns)
        value3 = st.selectbox("Select column 3", df.columns)
        # fig, ax = plt.subplots()
        # ax.bar(df.index, df[value])
        # st.pyplot(fig)
        chart_data = pd.DataFrame({
            f"{value1}": df[value1],
            f"{value2}": df[value2],
            f"{value3}": df[value3],
        })
        st.bar_chart(chart_data)

        # plot_str = "Bar plot of {}".format(value)
        # query = "Generate plot"
        # result = {"answer": plot_str, "plot": fig}
        # if 'history' not in st.session_state:
        #     st.session_state['history'] = []
        # st.session_state['history'].append((query, str(result)))


    # func for streamlit chat takes query from User
    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        print(result)
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"] 


    # appending history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Start message, in context of no question having being not asked yet
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! i'm Zobo (llama 2 model),now you can ask me anything related to" + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey u good! 👋"]

    # container for the chat history
    response_container = st.container() # form

    # container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Click here to interact (:", key='input') # user input values are here
            submit_button = st.form_submit_button(label='Send') # button to retrieve answer

        if submit_button and user_input:
            output = conversational_chat(user_input)
            if output is not None:
                st.session_state['past'].append(user_input) # old user input is appended
                st.session_state['generated'].append(output) # append the generated
            else:
                st.session_state['past'].append(user_input) # old user input is appended
                st.session_state['generated'].append('') # append an empty string if output is None

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")




