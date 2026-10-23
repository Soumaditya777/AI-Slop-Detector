import streamlit as st
import pandas as pd

from detector import SlopDetector, DetectorError

st.set_page_config(
    page_title="AI Slop Detector",
    page_icon="🔎",
    layout="wide",
)

MODEL_ID = "gouwsxander/slop-detector-bert"


@st.cache_resource(show_spinner="Loading the model for the first time…")
def load_detector():
    return SlopDetector(model_id=MODEL_ID)


st.title("🔎 AI Slop Detector")
st.write(
    "Estimate whether a passage resembles AI-generated text using a "
    "fine-tuned BERT classifier."
)

with st.expander("About this detector"):
    st.markdown(
        """
        - **Model:** [`gouwsxander/slop-detector-bert`](https://huggingface.co/gouwsxander/slop-detector-bert)
        - **Dataset:** [`gouwsxander/wikipedia-human-ai`](https://huggingface.co/datasets/gouwsxander/wikipedia-human-ai)
        - The model was trained on a particular text distribution. Performance can
          change substantially on other genres, models, languages, and writing styles.
        - This is a probabilistic estimate, **not proof of authorship**. Do not use
          it as the sole basis for academic, hiring, disciplinary, or other high-stakes decisions.
        """
    )

text = st.text_area(
    "Paste the text you want to analyze",
    height=260,
    placeholder="Paste one or more paragraphs here…",
    help="Long text is split into overlapping windows so more than one BERT context window can be analyzed.",
)

col_a, col_b = st.columns([1, 5])
with col_a:
    analyze = st.button("Analyze text", type="primary", use_container_width=True)
with col_b:
    st.caption(f"Characters: {len(text):,}")

if analyze:
    if not text.strip():
        st.warning("Paste some text before running the detector.")
        st.stop()

    try:
        detector = load_detector()
        with st.spinner("Analyzing text…"):
            result = detector.predict(text)
    except DetectorError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(
            "The model could not complete the analysis. Check your internet "
            "connection and package versions, then inspect the terminal for details."
        )
        st.exception(exc)
        st.stop()

    is_ai = result["predicted_label"] == "AI-generated"
    score = result["ai_probability"]

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Prediction")
        if is_ai:
            st.warning("The text resembles AI-generated text.")
        else:
            st.success("The text resembles human-written text.")
        st.metric("AI-generated score", f"{score * 100:.1f}%")
        st.progress(float(max(0.0, min(1.0, score))))
        st.caption(
            "This is the model's score for its AI class, not a calibrated guarantee "
            "that the text was written by AI."
        )

    with right:
        st.subheader("Analysis details")
        st.metric("Text windows analyzed", result["chunk_count"])
        st.metric("Approx. input tokens", f'{result["token_count"]:,}')
        st.metric("Processing device", result["device"])

    st.subheader("Per-window scores")
    chunks_df = pd.DataFrame(result["chunks"])
    if not chunks_df.empty:
        chunks_df["AI probability (%)"] = (chunks_df["ai_probability"] * 100).round(1)
        chunks_df["Prediction"] = chunks_df["ai_probability"].apply(
            lambda p: "AI-like" if p >= 0.5 else "Human-like"
        )
        st.dataframe(
            chunks_df[["chunk", "tokens", "AI probability (%)", "Prediction"]],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("View analyzed text windows"):
            for item in result["chunks"]:
                st.markdown(
                    f'**Window {item["chunk"]}** — '
                    f'AI score: {item["ai_probability"] * 100:.1f}%'
                )
                st.write(item["text"])
                st.divider()

    st.caption(
        "Human writing can be flagged, and AI-generated writing can be missed. "
        "Short passages are especially difficult to classify."
    )

st.divider()
st.caption("Local inference through Streamlit. The first run downloads the model from Hugging Face.")
