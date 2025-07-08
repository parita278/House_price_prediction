import streamlit as st
import joblib
import pandas as pd
import numpy as np
 
# Set page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)
 
# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('Model/house_price_model.pkl', 'rb') as file:
            model = joblib.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'house_price_model.pkl' not found. Please ensure the file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
 
# Load model
model = load_model()
 
# App title and description
st.title("🏠 House Price Prediction App")
st.markdown("Enter the house details below to get a price prediction")
if model is not None:
    # Create two columns for input
    st.markdown("<h2 style='text-align: center;'>📊 House Features</h2>",unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        # Common house features - adjust these based on your model
        area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1500, key="area")
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, key="bedrooms")
        bathrooms = st.number_input("Number of Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5, key="bathrooms")
        stories = st.number_input("Number of Stories", min_value=1, max_value=10, value=2, key="stories")
        mainroad = st.selectbox("Main Road",['Yes','No'], key="mainroad")
        guestroom = st.selectbox("Guest Room",['Yes','No'], key="guestroom")
    with col2:
        basement = st.selectbox("Basement",['Yes','No'], key="basement")
        hotwaterheating = st.selectbox("Hot Water Heating",['Yes','No'], key="hotwaterheating")
        airconditioning = st.selectbox("Air Conditioning",['Yes','No'], key="airconditioning")
        parking = st.number_input("Parking Spots", min_value=0, max_value=5, value=2, key="parking")
        prefarea = st.selectbox("Preferred Area",['Yes','No'], key="prefarea")
        furnishingstatus = st.selectbox("Furnishing Status",['Unfurnished','Semi','Furnished'], key="furnishingstatus")
    if st.button("🔮 Predict House Price", type="primary"):
        try:
            # Prepare input data - adjust column names based on your model
            input_data = pd.DataFrame({
                'area': [area],
                'bedrooms': [bedrooms],
                'bathrooms': [bathrooms],
                'stories': [stories],
                'mainroad': [1 if mainroad=='Yes' else 0],
                'guestroom': [1 if guestroom=='Yes' else 0],
                'basement': [1 if basement=='Yes' else 0],
                'hotwaterheating': [1 if hotwaterheating=='Yes' else 0],
                'airconditioning': [1 if airconditioning=='Yes' else 0],
                'parking': [parking],
                'prefarea': [1 if prefarea=='Yes' else 0],
                'furnishingstatus': [0 if furnishingstatus=='Unfurnished' else 1 if furnishingstatus=='Semi' else 2]
                
                
                
            })
            # Make prediction
            prediction = model.predict(input_data)[0]
            # Display results
            st.success("Prediction Complete!")
            # Create columns for results
            result_col1, result_col2 = st.columns(2)
            with result_col1:
                st.metric("💰 Predicted Price", f"${prediction:,.0f}")
            with result_col2:
                confidence_range = prediction * 0.1  # Assuming 10% confidence interval
                st.metric("📊 Confidence Range", f"±${confidence_range:,.0f}")
            # Additional insights
            st.subheader("🔍 Price Analysis")
            if prediction < 300000:
                st.info("💡 This property is in the lower price range for the area.")
            elif prediction < 600000:
                st.info("💡 This property is moderately priced for the area.")
            else:
                st.warning("💡 This property is in the higher price range for the area.")
            # Show input summary
            with st.expander("📋 Input Summary"):
                st.write("**Property Details Used for Prediction:**")
                for col, val in input_data.iloc[0].items():
                    st.write(f"- {col.replace('_', ' ').title()}: {val}")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check that your model expects the same feature names and data types.")
 
else:
    st.warning("Model could not be loaded. Please check the model file.")
 
# Footer
st.markdown("---")
st.markdown("**Note:** This prediction is based on historical data and should be used as a rough estimate only.")
 
# Sidebar with model info
with st.sidebar:
    st.header("ℹ️ Model Information")
    if model is not None:
        st.success("✅ Model loaded successfully")
        try:
            # Try to get model info if available
            if hasattr(model, 'feature_names_in_'):
                st.write("**Features used:**")
                for feature in model.feature_names_in_:
                    st.write(f"- {feature}")
        except:
            pass
    else:
        st.error("❌ Model not loaded")
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("1. Fill in the house details")
    st.markdown("2. Click 'Predict House Price'")
    st.markdown("3. View the estimated price")
    st.markdown("---")
    st.markdown("**Tips:**")
    st.markdown("- Ensure all fields are filled accurately")
    st.markdown("- Check that your model file is named 'house_price_model.pkl'")
    st.markdown("- Adjust feature names in code if needed")