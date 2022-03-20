import ast
import json
import logging
import zipfile
from io import BytesIO
from collections import Counter

import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def download_data_sample(api_url, token):
    try:
        # Set the path for eval API
        eval_url = api_url + "/prod/eval"
        
        # Set the authorization based on query parameter 'token', 
        # it is obtainable once you logged in to the modelshare website
        headers = {
            "Content-Type": "application/json", 
            "authorizationToken": token,
        }

        # Set the body indicating we want to get sample data from eval API
        data = {
            "exampledata": "TRUE"
        }
        data = json.dumps(data)

        # Send the request
        sample_tabulars = requests.request("POST", eval_url, 
                                           headers=headers, data=data).json()

        # Parsing the tabular data
        tabulars = json.loads(sample_tabulars['exampledata'])
        df = pd.DataFrame(np.array([tabulars['data']]),
                          columns=tabulars['columns'])

        # Prepare the data sample in csv
        csv_data = df.to_csv().encode('utf-8')
        
        # Setup a download button
        btn = st.download_button(
            label="Download data sample",
            data=csv_data,
            file_name="tabular_sample.csv",
            mime="text/csv"
        )
    except Exception as e:
        logging.error(e)

def display_result(data_frame, labels, statuses):
    status_label = {
        True: "Success",
        False: "Failed",
    }
    data_frame = data_frame.assign(status=[status_label[x] for x in statuses])
    data_frame = data_frame.assign(label=labels)
    st.table(data_frame)

    # Prepare the data sample in csv
    csv_data = data_frame.to_csv().encode('utf-8')

    # Setup a download button
    btn = st.download_button(
        label="Download result",
        data=csv_data,
        file_name="export.csv",
        mime="text/csv"
    )

def display_pie_chart(sizes, labels):
    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes)])
    st.plotly_chart(fig, use_container_width=True)
    
def display_bar_chart(freqs, labels):
    fig = px.bar(x=labels, y=freqs)
    st.plotly_chart(fig, use_container_width=True)
    
def display_stats(labels):
    counter = Counter(labels)
    unique_labels = list(counter.keys())
    freqs = list(counter.values()) # frequency of each labels

    # Size or portion in pie chart
    sizes = [float(x) / sum(freqs) * 100 for x in freqs]

    display_pie_chart(sizes, unique_labels)
    display_bar_chart(freqs, unique_labels)

def main():
    # Set the API url accordingly based on AIModelShare Playground API.
    api_url = "https://n0l8kcy3wh.execute-api.us-east-1.amazonaws.com"

    # Get the query parameter
    params = st.experimental_get_query_params()
    token = params['token'][0]

    st.header("Titanic Survival Classification")

    uploaded_file = st.file_uploader(
        label="Choose one csv and get the prediction",
        type=["csv"],
        accept_multiple_files=False,
    )

    download_data_sample(api_url, token)
        
    if uploaded_file:
        # Prepare the uploaded csv into per row record in json
        labels = []
        statuses = []
        data_frame = pd.read_csv(uploaded_file)
        for row in data_frame.itertuples():
            try:
                data = {}
                for col in data_frame.columns:
                    data[col] = [row[col]]

                data = json.dumps({"data": data})

                # Set the path for prediction API
                pred_url = api_url + "/prod/m"
                
                # Set the authorization based on query parameter 'token', 
                # it is obtainable once you logged in to the modelshare website
                headers = {
                    "Content-Type": "application/json", 
                    "authorizationToken": token,
                }

                # Send the request
                prediction = requests.request("POST", pred_url, 
                                              headers=headers, data=data)

                # Parse the prediction
                label = ast.literal_eval(prediction.text)[0]
                
                # Insert the label into labels
                labels.append(label)
                
                # Insert the API call status into statuses
                statuses.append(True)
            except Exception as e:
                logging.error(e)

                # add label as None if necessary
                if len(labels) < len(data_frame.shape[0]):
                    labels.append(None)
                statuses.append(False)

        display_result(data_frame, labels, statuses)
        display_stats(labels)

if __name__ == "__main__":
    main()