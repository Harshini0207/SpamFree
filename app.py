
import streamlit as st
import joblib

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="SpamFree | Email Spam Detector",
    
    layout="centered"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load("model/spam_detector.pkl")

model = load_model()

# ---------------- SESSION STATE ----------------

if "email_input" not in st.session_state:
    st.session_state.email_input = ""

def clear_input():
    st.session_state.email_input = ""

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #79baec;
}

/* Main content */
.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 2rem;
    color: Black;
}

/* Heading */
.main-title {
    color: #102A43 !important;
    font-size: 54px;
    font-weight: 800;
    text-align: center;
    margin-top: 45px;
    margin-bottom: 10px;
}

.subtitle {
    color: Black;
    text-align: center;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Section heading */
.section-title {
    color: Black;
    font-size: 23px;
    font-weight: 700;
}

/* Email input box */
.stTextArea textarea {
    background-color: #E7E7E7 !important;
    color: #000000 !important;
    border: 1px solid #CBD5E1;
    border-radius: 12px;
    font-size: 15px;
}

/* Placeholder text */
.stTextArea textarea::placeholder {
    color: #000000 !important;
    opacity: 1 !important;
}
/* Hover effect */
.stButton > button:hover {
    background-color: #174EA6 !important;
    color: white !important;
    border: 2px solid #174EA6 !important;
}
/* Footer */
.footer {
    text-align: center;
    color: #64748B;
    font-size: 13px;
    margin-top: 30px;
}
/* Button background and text */
.stButton button {
    background-color: #102A43 !important;
    color: #FFFFFF !important;
}

/* Button text */
.stButton button p {
    color: #FFFFFF !important;
}

/* Hover effect */
.stButton button:hover {
    background-color: #174EA6 !important;
    color: #FFFFFF !important;
}

.stButton button:hover p {
    color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">SpamFree</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Email Spam Detection'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- INTRODUCTION ----------------

st.markdown(
    '<div class="section-title">Check Your Email</div>',
    unsafe_allow_html=True
)

st.write(
    "Paste your email message below. "
    "Our machine learning model will analyze "
    "the content and classify it."
)

st.write("")

# ---------------- EMAIL INPUT ----------------

email_text = st.text_area(
    "Email Content",
    height=220,
    placeholder="Paste your email message here...",
    key="email_input"
)

# ---------------- BUTTONS ----------------

col1, col2 = st.columns(2)

with col1:
    check_button = st.button(
        "Check Email",
        use_container_width=True
    )

with col2:
    st.button(
        " Clear",
        use_container_width=True,
        on_click=clear_input
    )

# ---------------- PREDICTION ----------------

if check_button:

    if not email_text.strip():

        st.warning(
            "Please paste an email before checking."
        )

    else:

        prediction = model.predict([email_text])[0]

        probabilities = model.predict_proba([email_text])[0]

        classes = list(model.classes_)

        spam_index = classes.index(1)
        ham_index = classes.index(0)

        spam_probability = probabilities[spam_index] * 100
        ham_probability = probabilities[ham_index] * 100

        st.divider()

        st.markdown(
            '<div class="section-title">Analysis Result</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:

            st.error(
                "🚨 SPAM EMAIL DETECTED!"
            )

            st.write(
                "This email has been classified as Spam."
            )

        else:

            st.success(
                "✅ NOT SPAM"
            )

            st.write(
                "This email has been classified as legitimate."
            )

        # Probability details
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Spam Probability",
                f"{spam_probability:.2f}%"
            )

        with result_col2:
            st.metric(
                "Not Spam Probability",
                f"{ham_probability:.2f}%"
            )

        st.caption(
            "These are model-estimated probabilities, "
            "not a guarantee of email safety."
        )

# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    '<div class="footer">'
    'SmartMail | Powered by Multinomial Naive Bayes'
    '<br>Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)
