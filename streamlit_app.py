# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Streamlit app for univariate time series forecasting
# Reference: https://gitlab.scania.com/ixad/scania-prophet-time-series/-/blob/main/project/app.py
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import add_changepoints_to_plot, plot_plotly

from prophet.diagnostics import cross_validation, performance_metrics

st.set_page_config(page_title="Forecasting")

st.title("Forecasting")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Upload CSV data
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)


    st.write(df.head())
    time_column = st.selectbox(
    'Choose time column name: ',
    tuple(['<select>'] + list(df.columns)))
    

    forecasting_column = st.selectbox(
    'Choose forecasting column name: ',
    tuple(['<select>'] + list(df.columns)))
    if time_column!='<select>' and forecasting_column!='<select>':
        df = df[[time_column, forecasting_column]].copy()
        df.rename(columns={time_column:'ds',forecasting_column:'y'},inplace=True)

        weeks = st.selectbox(
        'How many weeks in the future do you want to do forecast: ',
        tuple(['<select>'] + [i for i in range(1,105)]))

        interval_width = st.selectbox('Choose interval width: ',
                                 tuple(['<select>'] + [i/100 for i in range(0,100,5)]))

        if weeks != '<select>' and interval_width!='<select>':

            model = Prophet(interval_width=float(interval_width))
            model.fit(df)

            future = model.make_future_dataframe(periods=int(weeks)*7)
            forecast = model.predict(future)

            tail_df = forecast.tail(int(weeks)*7)

            # Plot the forecast
            fig = plot_plotly(model,forecast)
            st.plotly_chart(fig,use_container_width=True)

            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')


            csv = convert_df(tail_df)

            st.download_button("Press to Download", csv, f"forecast_{forecasting_column}.csv", "text/csv", key='download-csv')
