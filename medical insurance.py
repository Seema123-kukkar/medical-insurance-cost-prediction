# ---------------------------------------------
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import mlflow.pyfunc
import joblib
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Medical Insurance Cost Prediction", layout="wide")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
app_mode = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📊 EDA",  "💰 Prediction", "🧠 MLflow Tracking"])




# ------------------- HOME -------------------
if app_mode == "🏠 Home":
    st.title("💰 Medical Insurance Cost Prediction")
    st.subheader("Predict Individual Health Insurance Charges")

    st.markdown("""
    **Project Goals:**
    - Predict medical insurance charges based on user data  
    - Understand how factors like age, BMI, smoking, and region affect cost  
    - Build an end-to-end regression pipeline and deploy it

    **Skills Applied:**  
    `Python` | `Pandas` | `Scikit-learn` | `Streamlit` | `MLflow` | `EDA`

    **Domain:**  
    Healthcare | Insurance | Data Science
    """)



elif app_mode == "📊 EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")

    uploaded_file = st.file_uploader("c:/Users/91997/Downloads/medical_insurance.csv", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("### What is the distribution of medical insurance charges?")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['charges'], kde=True, bins=30, color='teal', ax=ax)
        ax.set_title("Distribution of Medical Insurance Charges")
        ax.set_xlabel("Charges")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        

        st.write("### What is the age distribution of the individuals?")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['age'], kde=True, bins=30, color='red', ax=ax)
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        st.pyplot(fig)



        st.write("### How many people are smokers vs non-smokers?")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='smoker', data=df, palette='Set2', ax=ax)
        ax.set_title("Smokers vs Non-Smokers")
        ax.set_xlabel("Smoker Status")
        ax.set_ylabel("Count")
        st.pyplot(fig)

      
        st.write("### What is the average BMI in the dataset?")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['bmi'], kde=True, bins=30, color='green', ax=ax)
        ax.set_title("BMI Distribution")
        ax.set_xlabel("BMI")
        ax.set_ylabel("Count")
        st.pyplot(fig)


        
        st.write("### Which regions have the most number of policyholders?")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='region', data=df, palette='cool', ax=ax)
        ax.set_title("Number of Policyholders by Region")
        ax.set_xlabel("Region")
        ax.set_ylabel("Count")
        st.pyplot(fig)



        st.write("###  How do charges vary with age?")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x='age', y='charges', data=df, color='darkblue', ax=ax)
        ax.set_title("Charges vs Age")
        ax.set_xlabel("Age")
        ax.set_ylabel("Charges")
        st.pyplot(fig)



        st.write("### Is there a difference in average charges between smokers and non-smokers?")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='smoker', y='charges', data=df, ci=None, palette='Set2', ax=ax)
        ax.set_title("Average Charges by Smoking Status")
        ax.set_xlabel("Smoker (no / yes)")
        ax.set_ylabel("Average Charges")
        st.pyplot(fig)



        st.write("### Does BMI impact insurance charges?")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x='bmi', y='charges', data=df,hue='smoker',alpha=0.7,palette='Set1', ax=ax)
        ax.set_title("Medical Charges vs BMI")
        ax.set_xlabel("BMI")
        ax.set_ylabel("Medical Charges")
        st.pyplot(fig)


        
        st.write("### Do men or women pay more on average?")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='sex', y='charges', data=df, color='red', ax=ax)
        ax.set_title("Charges by Gender (Female vs Male)")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Charges")
        st.pyplot(fig)

        

        st.write("### Is there a correlation between the number of children and the insurance charges?")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='children', y='charges', data=df, palette='Set3', ax=ax)
        ax.set_title("Charges vs Number of Children")
        ax.set_xlabel("Number of Children")
        ax.set_ylabel("Insurance Charges")
        st.pyplot(fig)

        

        st.write("###  How does smoking status combined with age affect medical charges?")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            x='age',
            y='charges',
            hue='smoker',
            style='sex',
            data=df,
            palette='coolwarm',
            ax=ax
        )
        ax.set_title("Charges vs Age (Colored by Smoking Status, Styled by Gender)")
        ax.set_xlabel("Age")
        ax.set_ylabel("Medical Charges")
        st.pyplot(fig)




        st.write("### What is the impact of gender and region on charges for smokers?")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='region',y='charges',hue='sex', data=df,palette="coolwarm", ax=ax)
        ax.set_title("Impact of Gender and Region on Charges (for Smokers)")
        ax.set_xlabel("Region")
        ax.set_ylabel("Medical Charges")
        st.pyplot(fig)



        st.write("### How do age, BMI, and smoking status together affect insurance cost?")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x='age',y='charges',size='bmi', hue='smoker', alpha=0.7, sizes=(20, 100), palette='coolwarm', ax=ax)
        ax.set_title("Age vs Charges (Bubble Size = BMI, Color = Smoking Status)")
        ax.set_xlabel("Age")
        ax.set_ylabel("Insurance Charges")
        st.pyplot(fig)

 

        st.write("### Do obese smokers (BMI > 30) pay significantly higher than non-obese non-smokers?")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='smoker', y='charges', data=df, palette='Set2', ax=ax)
        ax.set_title("Medical Charges: Obese Smokers vs Non-Obese Non-Smokers")
        ax.set_xlabel("smoker")
        ax.set_ylabel("Insurance Charges")
        st.pyplot(fig)



        st.write("### Are there outliers in the charges column? Who are the individuals paying the highest costs?")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(y='charges', data=df,color='lightcoral', ax=ax)
        ax.set_title("Outlier Detection in Medical Charges")
        ax.set_ylabel("Insurance Charges")
        st.pyplot(fig)



        st.write("### What is the correlation between numeric features like age, BMI, number of children, and charges?")
        numeric_features = ['age', 'bmi', 'children', 'charges']
        corr_matrix = df[numeric_features].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Correlation between Numeric Features")
        st.pyplot(fig)

      
        
        st.write("### Which features have the strongest correlation with the target variable (charges)? ")
        corr = df.corr(numeric_only=True)['charges'].sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=corr.index, y=corr.values, palette='viridis', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylabel("Correlation with Charges")
        ax.set_title("Feature Correlation with Insurance Charges")
        st.pyplot(fig)
        


elif app_mode == "💰 Prediction":
    st.title("💰 Predict Medical Insurance Charges")

    st.markdown("Enter your details below to get a cost estimate:")


    MODEL_NAME = "GradientBoostingRegressor"  
    MODEL_STAGE = "Production"  
    LOCAL_MODEL_PATH = "best_model.pkl"  
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"  


    @st.cache_resource
    def load_model():
        try:
        # ✅ Set tracking URI first
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # ✅ Load from MLflow Model Registry
            model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            model = mlflow.pyfunc.load_model(model_uri)

            st.sidebar.success(f"✅ Model loaded from MLflow Registry: `{MODEL_STAGE}`")
            return model

        except Exception as e:
        # ⚠️ Fallback to local model if MLflow fails
            st.sidebar.warning(f"⚠️ MLflow load failed: {e}")
            if Path(LOCAL_MODEL_PATH).exists():
                model = joblib.load(LOCAL_MODEL_PATH)
                st.sidebar.info("✅ Loaded model from local fallback")
                return model
            else:
                st.sidebar.error("❌ No model found locally or on MLflow.")
                st.stop()

# ----------------------------
# 🧠 Load Model
# ----------------------------
    model = load_model()


# Example input fields
    age = st.number_input("Age", 18, 100, 30)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    children = st.number_input("Children", 0, 5, 1)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    sex = st.selectbox("Sex", ["male", "female"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Prepare input as DataFrame
    import pandas as pd
    input_data = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex_male": 1 if sex == "male" else 0,
    "smoker_yes": 1 if smoker == "yes" else 0,
    "region_northwest": 1 if region == "northwest" else 0,
    "region_southeast": 1 if region == "southeast" else 0,
    "region_southwest": 1 if region == "southwest" else 0,
}])

# ----------------------------
# 💵 Predict Button
# ----------------------------
    if st.button("💰 Predict Insurance Cost"):
        try:
            prediction = model.predict(input_data)
            cost = float(prediction[0])
            st.success(f"💵 Estimated Insurance Cost: **{cost:,.2f}**")
        except Exception as e:
             st.error(f"❌ Prediction failed: {e}")




# ------------------- MLFLOW TRACKING -------------------
elif app_mode == "🧠 MLflow Tracking":
    st.title("🧠 MLflow Experiment Tracking")
    st.markdown("""
    Track experiments, models, and metrics in **MLflow UI**.  
    open the link: [ http://127.0.0.1:5000]( http://127.0.0.1:5000)
    """)
