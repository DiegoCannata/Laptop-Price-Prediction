import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import os

st.markdown(
    """
    <style>
    [data-baseweb="tab-list"] {
        display: flex;
        width: 100%;
    }
    [data-baseweb="tab"] {
        width: 50% !important;
        flex: 1;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


try: 
    model = joblib.load(os.path.join("artifacts", "modelo_xgb.pkl"))  
    scaler = joblib.load(os.path.join("artifacts","scaler.pkl"))  
    transformer = joblib.load(os.path.join("artifacts","transformer.pkl"))  
    le_storage = joblib.load(os.path.join("artifacts","label_encoder_storage.pkl"))
except Exception as e:
    st.error(f"Error loading model or transformers: {e}")
    st.stop()

def convert_to_gb(storage):
    if 'TB' in storage:
        return float(storage.split('TB')[0].strip()) * 1024
    elif 'GB' in storage:
        return float(storage.split('GB')[0].strip())
    return 0

def split_resolution(resolution):
    width, height = resolution.split('x')
    return int(width), int(height)

def get_features(df):
    input_transformed = transformer.transform(df)
    columns_encoded = [col.split('__')[-1] for col in transformer.get_feature_names_out()]
    df_transformed = pd.DataFrame(input_transformed, columns=columns_encoded)
    return df_transformed


arr_brand = ["Dell", "HP", "Apple", "Asus", "Lenovo", "MSI", "Microsoft", "Acer", "Samsung", "Razer"]
arr_processor = ["Intel i3", "Intel i5", "Intel i7", "Intel i9", "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7", "AMD Ryzen 9"]
arr_ram = [4, 8, 16, 32, 64]
arr_storage = ["256GB SSD", "512GB SSD", "1TB HDD", "1TB SSD", "2TB SSD"]
arr_gpu = ["Nvidia RTX 2060", "Nvidia GTX 1650", "Nvidia RTX 3060", "AMD Radeon RX 6600", "AMD Radeon RX 6800", "Integrated", "Nvidia RTX 3080"]
arr_screen = [13.3, 14.0, 15.6, 16.0, 17.3]
arr_resolution = ["1366x768", "1920x1080", "2560x1440", "3840x2160"]
arr_ops = ["Windows", "macOS", "Linux", "FreeDOS"]

st.sidebar.header('Exploratory Data Analysis Filters')
side_brand = st.sidebar.multiselect("Brand", arr_brand, arr_brand)  
side_processor = st.sidebar.multiselect("Processor", arr_processor, arr_processor)
side_ram = st.sidebar.multiselect("RAM (GB)", arr_ram, arr_ram)
side_storage = st.sidebar.multiselect("Storage", arr_storage, arr_storage)
side_gpu = st.sidebar.multiselect("GPU", arr_gpu, arr_gpu)
side_screen_size = st.sidebar.multiselect("Screen Size (inch)", arr_screen, arr_screen)
side_resolution = st.sidebar.multiselect("Resolution", arr_resolution, arr_resolution)
side_battery_life = st.sidebar.slider("Battery Life (hours)", min_value=4, max_value=12, value=(4,12))
side_weight = st.sidebar.slider("Weight (kg)", min_value=1.2, max_value=3.5, value=(1.2,3.5))
side_ops = st.sidebar.multiselect("Operating System", arr_ops, arr_ops)

tab_prediction, tab_eda = st.tabs(['Predictions', 'EDA'])

with tab_prediction:
    st.title("Laptop Price Prediction 💻")
    st.markdown("""**Data source:** [Laptop price dataset](https://www.kaggle.com/datasets/armaanpreet123/laptop-price-dataset/data).""")
    st.image(os.path.join(os.getcwd(), "static", "laptop2.jpg"))

    with st.form(key="model_prediction_form"):
        st.subheader('Fill in the fields to find the price of your ideal laptop!')
        brand = st.selectbox("Brand", arr_brand)
        processor = st.selectbox("Processor", arr_processor)
        ram = st.selectbox("RAM (GB)", arr_ram)
        storage = st.selectbox("Storage", arr_storage)
        gpu= st.selectbox("GPU", arr_gpu)
        screen_size = st.selectbox("Screen Size (inch)", arr_screen)
        resolution = st.selectbox("Resolution", arr_resolution)
        battery_life = st.number_input("Battery life (hours)", min_value=4, max_value=12, value=8)
        weight = st.number_input("Weight (kg)", min_value=1.2, max_value=3.5, value=2.0)
        ops = st.selectbox("Operating System", arr_ops)

        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        storage_size = convert_to_gb(storage)
        storage_type = storage.split()[-1]
        width, height = split_resolution(resolution)
        input_df = pd.DataFrame([[brand, processor, ram, storage_size, gpu, screen_size, battery_life, weight, ops, storage_type, width, height]],
                                columns=["Brand", "Processor", "RAM (GB)", "Storage", "GPU", "Screen Size (inch)", "Battery Life (hours)", "Weight (kg)", "Operating System", "Storage Type", "Width", "Height"])
        input_df_transformed = get_features(input_df)
        input_df_transformed["Storage Type"] = le_storage.transform([storage_type])
        input_df_transformed = input_df_transformed.astype(float)
        X_scaled = scaler.transform(input_df_transformed)
        df_scaled = pd.DataFrame(X_scaled, columns=input_df_transformed.columns)
        try:
            prediction = model.predict(df_scaled)
            st.success(f"Estimated Price: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            
    st.info("Note: The model's prediction error is around $110.")
    st.write('---')
    st.header('Feature Importance')
    st.image(os.path.join(os.getcwd(), "static", "shape_plot.png"))
    st.write('---')
    st.header('Correlation')
    st.image(os.path.join(os.getcwd(), "static", "correlation.png"))

with tab_eda:
    df = pd.read_csv('laptop_prices.csv')

    df_selected = df[
    (df["Brand"].isin(side_brand)) &
    (df["Processor"].isin(side_processor)) &
    (df["RAM (GB)"].isin(side_ram)) &
    (df["Storage"].isin(side_storage)) &
    (df["GPU"].isin(side_gpu)) &
    (df["Screen Size (inch)"].isin(side_screen_size)) &
    (df["Resolution"].isin(side_resolution)) &
    (df["Operating System"].isin(side_ops)) &
    (df["Battery Life (hours)"] >= side_battery_life[0]) & (df["Battery Life (hours)"] <= side_battery_life[1]) &
    (df["Weight (kg)"] >= side_weight[0]) & (df["Weight (kg)"] <= side_weight[1])]

    st.header('Display data for selected laptops')
    st.dataframe(df_selected)
    st.write('Data Dimension: ' + str(df_selected.shape[0]) + ' rows and ' + str(df_selected.shape[1]) + ' columns.')
    csv = df_selected.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name='selected_data.csv', mime='text/csv')   

    st.header('Key things you should know before buying a computer')
    st.markdown('[Click here to view the analysis on Google Colab](https://colab.research.google.com/drive/14VN1c4HDu0rQbcEro_FelIAKcjjmV5-Q#scrollTo=jkGFDT-vNneV)', unsafe_allow_html=True)
    st.write('These are the most expensive brands on the market. We can see that MSI, Razer, and Apple stand out, with Apple being the priciest—its prices are 33.5% above the average.')
    st.image(os.path.join(os.getcwd(), "static", "avg_price_brand.png"))
    st.write("") 
    st.write('We can see that if you look for the newest processors like Intel i9 and AMD Ryzen 9 it will cost you 40% more than the average.')
    st.image(os.path.join(os.getcwd(), "static", "avg_price_processor.png"))
    st.write("") 
    st.write("If you're looking for a cheaper laptop, you can look for a computer with an integrated GPU which costs 31.2% less than the average.")
    st.image(os.path.join(os.getcwd(), "static", "avg_price_gpu.png"))
    st.write("") 
    st.write("One of the features that most affects the price is the RAM, here you can see how the average price increases each time the RAM increases")
    st.image(os.path.join(os.getcwd(), "static", "avg_price_ram.png"))
    st.write("") 
    st.write("Likewise, resolution significantly impacts the price of laptops. Choosing a laptop with the highest resolution will cost you 38.1% more than the average laptop.")
    st.image(os.path.join(os.getcwd(), "static", "avg_price_resolution.png"))

