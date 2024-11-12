import streamlit as st

def display_image_columns(processed_image, hand_region, prediction_and_confidence):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(processed_image, caption="Original", use_column_width=True)
    with col2:
        st.image(hand_region, caption="Hand region")
    with col3:
        prediction, confidence = prediction_and_confidence
        st.markdown(
            f"<p style='font-size:22px; font-weight:bold;'>{prediction}</p>",
            unsafe_allow_html=True
        )
        st.write(f"Confidence: {confidence}%")
