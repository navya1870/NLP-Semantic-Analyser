
import streamlit as st
import joblib
import pandas as pd


# -----------------------------------
# 1. LOAD TRAINED MODEL AND VECTORIZER
# -----------------------------------

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# -----------------------------------
# 2. EMOTION LABEL MAPPING
# -----------------------------------

emotion_labels = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

emotion_emoji = {
    "sadness": "😢",
    "anger": "😡",
    "love": "❤️",
    "surprise": "😲",
    "fear": "😨",
    "joy": "😊"
}


# -----------------------------------
# 3. PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="EmotionSense AI",
    page_icon="🧠",
    layout="wide"
)


# -----------------------------------
# 4. CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #172554);
    color: white;
}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

.hero {
    text-align: center;
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(120deg, #312e81, #1e40af);
    margin-bottom: 30px;
}

.hero h1 {
    color: white;
    font-size: 42px;
}

.hero p {
    color: #dbeafe;
    font-size: 17px;
}

div.stButton > button {
    background: linear-gradient(90deg, #6366f1, #3b82f6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-size: 17px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #4f46e5, #2563eb);
    color: white;
}

.result-card {
    background: #1e293b;
    border: 1px solid #475569;
    border-radius: 15px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
}

.result-card h2 {
    color: #93c5fd;
    font-size: 32px;
}

.result-card p {
    color: #cbd5e1;
    font-size: 16px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 50px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# 5. HEADER
# -----------------------------------

st.markdown("""
<div class="hero">
    <h1>🧠 EmotionSense AI</h1>
    <p>Understand the emotion behind every sentence</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------
# 6. INTRODUCTION
# -----------------------------------

st.markdown("### ✨ Emotion Analysis")

st.write(
    "Enter any sentence below and our Machine Learning "
    "model will identify the emotion expressed in it."
)


# -----------------------------------
# 7. USER INPUT
# -----------------------------------

user_text = st.text_area(
    "Your Text",
    placeholder="Example: I am really happy today!",
    height=130
)


# -----------------------------------
# 8. PREDICTION
# -----------------------------------

if st.button("Analyze Emotion 🔍"):

    if user_text.strip() == "":
        st.warning("Please enter some text first.")

    else:

        # Convert text into Bag of Words
        text_vector = vectorizer.transform([user_text])

        # Predict numerical emotion
        prediction = model.predict(text_vector)[0]

        # Convert number back into emotion name
        emotion = emotion_labels[int(prediction)]

        # Get corresponding emoji
        emoji = emotion_emoji.get(emotion, "💬")

        # Display result
        st.markdown("---")
        st.markdown("### 🎯 Analysis Result")

        st.markdown(f"""
        <div class="result-card">
            <h2>{emoji} {emotion.capitalize()}</h2>
            <p>Emotion detected by our AI model</p>
        </div>
        """, unsafe_allow_html=True)


        # -----------------------------------
        # 9. EMOTION PROBABILITIES
        # -----------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(text_vector)[0]

            # Convert numerical classes into emotion names
            probability_df = pd.DataFrame({
                "Emotion": [
                    emotion_labels[int(label)]
                    for label in model.classes_
                ],
                "Probability": probabilities * 100
            })

            probability_df = probability_df.sort_values(
                by="Probability",
                ascending=False
            )

            confidence = max(probabilities) * 100

            # Metrics
            st.write("")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

            with col2:
                st.metric(
                    "Emotions Identified",
                    len(probabilities)
                )


            # -----------------------------------
            # 10. PROBABILITY CHART
            # -----------------------------------

            st.markdown("### 📊 Emotion Probability Distribution")

            chart_df = probability_df.set_index("Emotion")

            st.bar_chart(chart_df)


            # -----------------------------------
            # 11. DETAILED ANALYSIS
            # -----------------------------------

            st.markdown("### 📋 Detailed Analysis")

            display_df = probability_df.copy()

            display_df["Probability"] = (
                display_df["Probability"].round(2).astype(str) + "%"
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


# -----------------------------------
# 12. FOOTER
# -----------------------------------

st.markdown("""
<div class="footer">
    Built with Python | Logistic Regression |
    Bag of Words | Streamlit
    <br><br>
    EmotionSense AI © 2026
</div>
""", unsafe_allow_html=True)
