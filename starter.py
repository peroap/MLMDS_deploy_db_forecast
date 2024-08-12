import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from supabase import create_client, Client
from statsforecast import StatsForecast
from statsforecast.models import CrostonOptimized

# Initialize connection to db
@st.cache_resource
def init_connection():
    url:str=st.secrets["supabase_url"]
    api_key:str=st.secrets["supabase_api_key"]
    client:Client = create_client(url, api_key)
    return client

# Run the function to make the connection
supabase = init_connection()

# Function to query the db
# Return all data
@st.cache_resource(ttl=600)  # cache clears after 10 minutes
def run_query():
    return supabase.table('car_parts_monthly_sales').select("*").execute()

# Function to create a Dataframe
# Make sure that volume is an integer
# Return dataframe
@st.cache_data(ttl=600)
def create_dataframe()->pd.DataFrame:
    rows = run_query()
    df = pd.json_normalize(rows.data)
    # drop 4/1/2000 for part_id==2673 since all other part ids don't have any entry for it
    df = df[df["date"]!="4/1/2002"]
    return df

# Function to plot data
@st.cache_data
def plot_volume(ids)->None:
    fig, ax = plt.subplots()

    df['volume'] = df['volume'].astype(int)

    x = df[df["parts_id"] == 2674]['date']

    for id in ids:
        ax.plot(x,
                df[df['parts_id'] == id]['volume'], label=id)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.legend(loc='best')
    fig.autofmt_xdate()

    st.pyplot(fig)
    return None

# Function to format the dataframe as expected
# by statsforecast
@st.cache_data
def format_dataset(ids)->pd.DataFrame:
    model_df = df[df['parts_id'].isin(ids)]
    model_df = model_df.drop(['id'], axis=1)
    model_df.rename({"parts_id": "unique_id", "date": "ds",
                    "volume": "y"}, axis=1, inplace=True)
    model_df["y"] = pd.to_numeric(model_df["y"])
    return model_df

# Create the statsforecast object to train the model
# Return the statsforecast object
@st.cache_resource
def create_sf_object(model_df)->StatsForecast:
    sf = StatsForecast(
    df=model_df,
    models = [CrostonOptimized()],
    freq = 'MS'
    )
    return sf

# Function to make predictions
# Inputs: product_ids and horizon
# Returns a CSV
@st.cache_data(show_spinner="Making predictions...")
def make_predictions(ids, horizon):
    model_df = format_dataset(ids)
    trained_sf = create_sf_object(model_df)
    forecast_df = trained_sf.forecast(h=horizon)
    return forecast_df.to_csv(header=True)

if __name__ == "__main__":
    st.title("Forecast product demand")

    df = create_dataframe()

    st.subheader("Select a product")
    product_ids = st.multiselect(
        "Select product ID", options=df['parts_id'].unique())
    st.write(product_ids)

    plot_volume(product_ids)

    with st.expander("Forecast"):
        if len(product_ids) == 0:
            st.warning("Select at least one product ID to forecast")
        else:
            horizon = st.slider("Horizon", 1, 12, step=1)

            forecast_btn = st.button("Forecast", type="primary")

            # Download CSV file if the forecast button is pressed
            if forecast_btn:
                csv_file = make_predictions(product_ids, horizon)
                ids_for_file_name = "_".join((str(id_i) for id_i in product_ids))
                st.download_button(
                    label="Download predictions",
                    data=csv_file,
                    file_name=f"forecast_{ids_for_file_name}_{horizon}.csv",
                    mime="text/csv")