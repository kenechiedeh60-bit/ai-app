import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="My AI App")
st.title("SENTIMENT AND WORD GUESSING APP")

@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )

@st.cache_resource
def load_fill_mask_model():
    return pipeline("fill-mask", model="bert-base-uncased")


tab1, tab2 = st.tabs(["Sentiment", "Missing Word"])

with tab1:
    st.write("Type sentence. The model will say if it is POSITIVE or NEGATIVE.")
    text = st.text_area("Your sentence", "I love learning about artificial intelligence")

    if st.button("Check sentiment"):
        if not text.strip():
            st.warning("please type a sentence first.")
        else:
            with st.spinner("Thinking..."):
                result = load_sentiment_model()(text, truncation=True)[0]
                message = f"{result['label']} (score: {round(result['score'], 3)})"
                if result["label"] == "POSITIVE":
                    st.success(message)
                else:
                    st.error(message)