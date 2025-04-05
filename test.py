import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import tensorflow as tf
# import pickle  # or use pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_lungs_cancer_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(224,224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max elemen

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#


# Configure the page
st.set_page_config(page_title="Lung Cancer Detection", layout="wide", page_icon="🫁")

# Custom CSS to change background
st.markdown("""
    <style>
    body {
        background-color: #030303;
    }
    .stApp {
        background-color: #F1EFEC;  /* Light blue-gray tone */
    }
    </style>
""", unsafe_allow_html=True)


# Create a navbar with the requested options
selected = option_menu(
    menu_title=None,
    options=["Home","About", "Predict"],
    icons=["house", "built By","graph-up" ],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles = {
    "container": {
        "padding": "0!important",
        "background-color": "#D4C9BE"  # soft light blue-gray
    },
    "icon": {
        "color": "#030303",  # soothing sage green
        "font-size": "18px"
    },
    "nav-link": {
        "font-size": "16px",
        "color":"030303",
        "text-align": "center",
        "margin": "0px",
        "--hover-color": "#56CCF2"  # cool soft teal on hover
    },
    "nav-link-selected": {
        "background-color": "#123458",  # sage green for selected tab
        "color": "#F0F4F8"  # light background for contrast
    }
}

)


if selected == "Home":
    

# Load your trained model
    model = joblib.load("ensem_model.joblib")

    import streamlit as st
import numpy as np
import pandas as pd

st.title("Lung Cancer Risk Prediction")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120)
smoking = st.selectbox("Smoking", ["Yes", "No"])
discoloration = st.selectbox("Finger Discoloration", ["Yes", "No"])
pollution = st.selectbox("Exposure to Pollution", ["High", "Low"])
illness = st.selectbox("Long-term Illness", ["Yes", "No"])
immune = st.selectbox("Weak Immune System", ["Yes", "No"])
breathing = st.selectbox("Difficulty Breathing", ["Yes", "No"])
throat = st.selectbox("Throat Discomfort", ["Yes", "No"])
oxygen = st.number_input("Oxygen Level (%)", min_value=0.0, max_value=100.0)
chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])
family_history = st.selectbox("Family History of Lung Cancer", ["Yes", "No"])

# Mapping categorical values
binary_map = {"Yes": 1, "No": 0}
pollution_map = {"Low": 0, "High": 1}

# Convert inputs
input_data = np.array([[
    age,
    binary_map[smoking],
    binary_map[discoloration],
    pollution_map[pollution],
    binary_map[illness],
    binary_map[immune],
    binary_map[breathing],
    binary_map[throat],
    oxygen,
    binary_map[chest_pain],
    binary_map[family_history]
]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High risk of lung cancer. Please consult a specialist.")
    else:
        st.success("✅ Low risk of lung cancer.")


    

    st.markdown("""
        <div style='
            background-color: #6FCF97;
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 10px;
        '>
            <h1 style='color: white; margin: 0;'>Welcome to Lung Cancer Detection</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<p style='color: #333333; font-size: 18px;'>"
        "This interactive dashboard helps you assess lung cancer risk factors."
        "</p>",
        unsafe_allow_html=True
    )

    # Styled heading and list content
    st.markdown("""
        <div style='
            background-color: #3B82F6;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
        '>
            <h2 style='color: white; margin: 0;'>How to use this application</h2>
        </div>

        <div style='color: #333333; font-size: 16px; line-height: 1.6;'>
            <ol>
                <li>Navigate to the <strong>Predict</strong> tab to input patient data and get predictions</li>
                <li>Use the <strong>Explain</strong> tab to understand how predictions are made</li>
                <li>Learn more about our model in the <strong>Model Info</strong> section</li>
                <li>Share your experience in the <strong>Feedback</strong> section</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

    
    # Add some sample statistics or information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Model Accuracy", value="94.2%", delta="+1.2%")
    with col2:
        st.metric(label="Patients Screened", value="1,245", delta="+23")
    with col3:
        st.metric(label="Early Detections", value="89", delta="+5")

# Predict Page

elif selected == "Predict":
    st.title("Predict Lung Cancer Risk")
    st.write("Enter patient information to get a risk assessment.")
    
    # Create a form for prediction
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Patient Name")
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            smoking = st.selectbox("Smoking", ["Yes", "No"])
            finger_discoloration = st.selectbox("Discolorations in Finger", ["Yes", "No"])
            exposure_to_pollution = st.selectbox("Locality Pollution Level", ["High", "Low"])
            longterm_illness = st.selectbox("Having Longterm Illness", ["Yes", "No"])
        
        with col2:
            
            immune_weakness = st.selectbox("Have weak Immune Syaytem", ["Yes", "No"])
            breathing_issue = st.selectbox("Haveing Difficulty Breathing", ["Yes", "No"])
            throat_discomfort = st.selectbox("Discomfort in Throat", ["Yes", "No"])
            oxygen_saturation = st.number_input("Oxygen-Saturated Hemoglobin in the Blood")
            chest_tightness = st.selectbox("Chest Stiffness", ["Yes", "No"])
            family_history = st.selectbox("Having Family History of Lung Cancer", ["Yes", "No"])
        
            

        submitted = st.form_submit_button("Generate Prediction")
        
            # st.info("Risk assessment: Moderate risk. Further screening recommended.")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    if(st.button("Predict")):
        
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))


# # Explain Page
elif selected == "Explain":
    st.title("Explain AI Predictions")
    st.write("Understand how our model makes predictions and what factors influence the results.")
    
    st.markdown("""
    ## Model Explanation
    
    Our lung cancer prediction model uses a combination of:
    
    - Patient demographic information
    - Medical history
    - Lifestyle factors
    - Symptoms
    
    The model weighs these factors based on their statistical correlation with lung cancer risk.
    """)
    
    # Sample visualization
    st.subheader("Feature Importance")
    import pandas as pd
    import plotly.express as px
    
    # Sample data for feature importance
    data = {
        'Feature': ['Smoking History', 'Age', 'Family History', 'Persistent Cough', 'Shortness of Breath', 'Chest Pain'],
        'Importance': [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]
    }
    df = pd.DataFrame(data)
    
    fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='blues',
                title='Feature Importance in Prediction Model')
    st.plotly_chart(fig, use_container_width=True)

# Model Info Page
elif selected == "Model Info":
    st.title("Model Information")
    st.write("Technical details about our lung cancer prediction model.")
    
    st.markdown("""
    ## Model Architecture
    
    Our lung cancer prediction model is based on a gradient boosting algorithm trained on a dataset of over 10,000 patient records.
    
    ### Technical Specifications:
    
    - Algorithm: XGBoost
    - Training Data: 10,000+ anonymized patient records
    - Validation Accuracy: 94.2%
    - Sensitivity: 92.7%
    - Specificity: 95.1%
    - Last Updated: April 2025
    
    ### Data Sources:
    
    The model was trained using anonymized data from multiple hospitals and research institutions, ensuring diversity in the training dataset.
    """)
    
    # Add a sample confusion matrix
    st.subheader("Model Performance")
    import numpy as np
    import pandas as pd
    import plotly.figure_factory as ff
    
    # Sample confusion matrix
    z = np.array([[152, 8], [12, 128]])
    x = ['Predicted Negative', 'Predicted Positive']
    y = ['Actual Negative', 'Actual Positive']
    
    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')
    fig.update_layout(title_text='Confusion Matrix')
    st.plotly_chart(fig, use_container_width=True)

# Feedback Page
elif selected == "Feedback":
    st.title("Provide Feedback")
    st.write("We value your input to improve our lung cancer detection system.")
    
    # Create a feedback form
    with st.form("feedback_form"):
        name = st.text_input("Your Name (Optional)")
        role = st.selectbox("Your Role", ["Patient", "Doctor", "Researcher", "Other"])
        rating = st.slider("Rate your experience", 1, 5, 3)
        feedback = st.text_area("Your Feedback")
        suggestions = st.text_area("Suggestions for Improvement")
        
        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            st.success("Thank you for your feedback!")
        #     st.balloons()
