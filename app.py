import streamlit as st
import pickle
import numpy as np

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

# Helper function for Indian Currency Formatting (Lakhs and Crores)
def format_indian_currency(num):
    s, *d = str(num).partition(".")
    r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
    return "".join([r] + d)

# 2. Cache the model loading so it doesn't slow down the app on every interaction
@st.cache_resource
def load_models():
    pipe = pickle.load(open('pipe.pkl','rb'))
    df = pickle.load(open('df.pkl','rb'))
    return pipe, df

pipe, df = load_models()

# 3. Header Section
st.title("💻 Laptop Price Predictor")
st.markdown("<p style='font-size: 18px; color: gray;'>Configure your desired specifications below to estimate the market price.</p>", unsafe_allow_html=True)
st.divider()

# 4. Layout Generation: Grouping inputs logically

st.subheader("🛠️ Basic Information")
col1, col2, col3 = st.columns(3)
with col1:
    company = st.selectbox('Brand', df['Company'].unique())
with col2:
    type = st.selectbox('Type', df['TypeName'].unique())
with col3:
    os = st.selectbox('Operating System', df['os'].unique())

st.subheader("📺 Display Specifications")
col4, col5, col6, col7 = st.columns(4)
with col4:
    screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.0)
with col5:
    resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
with col6:
    touchscreen = st.radio('Touchscreen Feature', ['No','Yes'], horizontal=True)
with col7:
    ips = st.radio('IPS Panel', ['No','Yes'], horizontal=True)

st.subheader("⚡ Performance & Storage")
col8, col9, col10 = st.columns(3)
with col8:
    cpu = st.selectbox('CPU Processor', df['Cpu brand'].unique())
    gpu = st.selectbox('GPU Graphics', df['Gpu brand'].unique())
with col9:
    ram = st.selectbox('RAM (in GB)', [2,4,6,8,12,16,24,32,64])
    weight = st.number_input('Weight of the Laptop (kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1)
with col10:
    ssd = st.selectbox('SSD (in GB)', [0,8,128,256,512,1024])
    hdd = st.selectbox('HDD (in GB)', [0,128,256,512,1024,2048])

st.divider()

# 5. Prediction Action
# Center the button using columns
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
with btn_col2:
    predict_btn = st.button('Predict Price 🚀', use_container_width=True)

if predict_btn:
    # Preprocess the data
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
    
    # Format the query for the model
    query = np.array([company, type, ram, weight, touchscreen_val, ips_val, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1,12)
    
    # Make the prediction
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    
    # 6. Displaying the Result 
    st.balloons()
    
    # Format the number with Indian commas
    formatted_price = format_indian_currency(predicted_price)
    
    st.success("Analysis Complete!")
    st.metric(label="Estimated Configuration Price", value=f"₹ {formatted_price}")