
import streamlit as st
import joblib

# Load trained models
service_model = joblib.load("models/service_model.pkl")
priority_model = joblib.load("models/priority_model.pkl")

# Page configuration
st.set_page_config(
    page_title="AI Lead Intelligence Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Lead Intelligence Assistant")
st.write(
    "Analyze customer enquiries, identify service requirements, "
    "estimate lead priority, and recommend the next action."
)

st.info(
    "Prototype notice: This application uses a small synthetic dataset. "
    "Predictions are experimental and should be reviewed by a human."
)

# Recommended actions
actions = {
    "AI Development": "Arrange a discussion to understand the AI use case and data requirements.",
    "Web Development": "Schedule a website requirements discussion and prepare an estimate.",
    "App Development": "Collect platform, feature, and budget requirements.",
    "Cloud Services": "Assess infrastructure requirements and suggest a cloud solution.",
    "Digital Marketing": "Discuss campaign goals, target audience, and marketing budget."
}

priority_actions = {
    "High": "Contact the enquiry promptly for further qualification.",
    "Medium": "Follow up and collect additional project requirements.",
    "Low": "Share relevant information and schedule a follow-up."
}

# Input section
st.subheader("Enter Customer Enquiry")

enquiry = st.text_area(
    "Customer message",
    placeholder="Example: We need an AI chatbot for our customer support team.",
    height=130
)

if st.button("Analyze Enquiry", type="primary"):
    if not enquiry.strip():
        st.warning("Please enter a customer enquiry.")
    else:
        # Predict service and priority
        service = service_model.predict([enquiry])[0]
        priority = priority_model.predict([enquiry])[0]

        # Calculate prediction confidence
        service_confidence = max(
            service_model.predict_proba([enquiry])[0]
        ) * 100

        priority_confidence = max(
            priority_model.predict_proba([enquiry])[0]
        ) * 100

        # Display results
        st.subheader("Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Service", service)
            st.metric(
                "Service Confidence",
                f"{service_confidence:.1f}%"
            )

        with col2:
            st.metric("Estimated Priority", priority)
            st.metric(
                "Priority Confidence",
                f"{priority_confidence:.1f}%"
            )

        st.subheader("Recommended Next Action")
        st.write(actions.get(service, "Review the service requirements."))
        st.write(priority_actions.get(priority, "Review the lead manually."))

        st.caption(
            "Confidence is the model's estimated probability, "
            "not a guarantee that the prediction is correct."
        )

        st.warning(
            "Please verify the predicted priority before making "
            "any business decision."
        )
