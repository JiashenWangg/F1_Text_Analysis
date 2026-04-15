import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="F1 Transcript Classifier", layout="wide")

st.title("F1 Transcript Classifier")
st.write("Paste a press-conference corpus below and get the model prediction and explanation.")

text = st.text_area("Corpus", height=300, placeholder="Paste transcript text here...")

if st.button("Predict"):
    if not text.strip():
        st.warning("Please paste some text first.")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": text},
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            st.write(result['explanation'][0])

            if "features" in result and result["features"]:
                st.subheader("Extracted Features")
                df = pd.DataFrame(result["features"])
                st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
