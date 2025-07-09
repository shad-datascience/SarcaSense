import streamlit as st
import keras
import model

st.markdown('This app is created by: <a href="https://shad-datascience.guthub.io" target="_blank">Shad Jamil</a>', unsafe_allow_html=True)
st.title("SARCA-SENSE(A Sarcasm Detection App)")
st.write("Enter a code-mixed hinglish sentence to check if it's sarcastic.")

text_input = st.text_input("Input Text:")

classifier = keras.models.load_model('./src/mlp_model.keras')

if st.button("Predict"):
    if text_input:
        transformed_text = model.feature_pipeline.transform([text_input])
        prediction = classifier.predict(transformed_text)

        if prediction >= 0.5:
            st.success("🌀 Prediction: **Sarcastic**")
        else:
            st.info("✅ Prediction: **Not Sarcastic**")
        st.write("Raw Prediction Score:", float(prediction[0][0]))
    else:
        st.warning("Please enter some text.")
