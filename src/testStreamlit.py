import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
from evaluate import load_model, predict_stance
import torch
from dataset import load_dataset
import os

# fake data
blindspot_data = [
    {"title": "New Tax Bill Introduced", "bias": "Right Blindspot", "url": "https://cnn.com/taxbill", "mentions": 50,
     "left": 5, "center": 20, "right": 25},
    {"title": "Foreign Policy Shift", "bias": "Left Blindspot", "url": "https://foxnews.com/foreignpolicy",
     "mentions": 40, "left": 3, "center": 15, "right": 22},
    {"title": "Climate Change Report Released", "bias": "Right Blindspot", "url": "https://bbc.com/climate",
     "mentions": 45, "left": 4, "center": 18, "right": 23},
]

bias_stats = {
    "Left": random.randint(30, 50),
    "Center": random.randint(20, 40),
    "Right": random.randint(30, 50)
}

source_bias_data = [
    {"source": "CNN", "bias": "Left"},
    {"source": "Fox News", "bias": "Right"},
    {"source": "BBC", "bias": "Center"},
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BEST_MODEL = os.path.join(BASE_DIR, "outputs", "sequential_20250129_185113", "model_after_2.pt")
MODEL_NAME = "roberta-base"
TEST_DATASETS = {
    "Test Set 1": os.path.join(BASE_DIR, "data", "processed", "test1.json"),
    "Test Set 2": os.path.join(BASE_DIR, "data", "processed", "test2.json"),
    "Test Set 3": os.path.join(BASE_DIR, "data", "processed", "test3.json")
}


LABEL_NAMES = {
    0: "far_right",
    1: "right",
    2: "center",
    3: "left",
    4: "far_left"
}

# Update mapping for dataset labels (-2 to 2) with JSON names
DATASET_LABEL_NAMES = {
    -2: "far_right",
    -1: "right",
    0: "center",
    1: "left",
    2: "far_left"
}


def load_test_data():
    """Load test data and create a clean version without labels"""
    data = load_dataset(TEST_DATA_PATH)
    display_data = []
    for item in data:
        # Format text with tags like [title] and [source]
        formatted_text = f"[title] {item['title']} [/title] "
        if 'mention_source' in item:
            formatted_text += f"[source] {item['mention_source']} [/source] "
        formatted_text += f"[content] {item['text']} [/content]"

        display_data.append({
            'text': formatted_text,
            'title': item['title'],
            'true_label': item['label'],
        })
    return display_data


def predict_batch(texts, model, tokenizer, device):
    """Batch prediction for efficiency"""
    predictions = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_preds = [predict_stance(text, model, tokenizer, device) for text in batch_texts]
        predictions.extend(batch_preds)
    return predictions


if "search_history" not in st.session_state:
    st.session_state["search_history"] = []

st.set_page_config(page_title="News Bias Dashboard", layout="wide")

page_options = ["📌 Blindspot Events", "📊 Bias Analysis", "🔎 Custom Search", "🤖 ML Inference"]
selected_page = st.sidebar.selectbox("Navigation", page_options, index=0)

if "selected_event" not in st.session_state:
    st.session_state["selected_event"] = None


def show_event_details(event):
    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("🔙 Back", key="back_button_top"):
            st.session_state["selected_event"] = None
            st.rerun()

    st.title(f"📊 {event['title']} Analysis")
    st.write(f"**Bias:** {event['bias']}")
    st.write(f"🔗 [Read more]({event['url']})")
    if "left" in event and "center" in event and "right" in event:
        st.markdown("### 📈 Media Coverage Distribution")
        event_stats = {"Left": event["left"], "Center": event["center"], "Right": event["right"]}
        st.bar_chart(event_stats)

    if st.button("🔙 Back to Blindspot Events", key="back_button_bottom"):
        st.session_state["selected_event"] = None
        st.rerun()


if st.session_state["selected_event"]:
    show_event_details(st.session_state["selected_event"])
else:
    if selected_page == "📌 Blindspot Events":
        st.title("🔍 Blindspot Events")
        st.markdown("### 📰 Latest Blindspots")

        selected_bias = st.radio("Filter by:", ["All", "Left Blindspot", "Right Blindspot"], horizontal=True)

        for event in blindspot_data:
            if selected_bias == "All" or event["bias"] == selected_bias:
                if st.button(f"📌 {event['title']}", key=event['title']):
                    st.session_state["selected_event"] = event
                    st.rerun()

        st.markdown("### 📊 Overall Blindspot Trends")
        st.bar_chart(bias_stats)

    elif selected_page == "📊 Bias Analysis":
        st.title("📊 Bias Analysis")

        view_mode = st.radio("Select View Mode:", ["Bar Chart", "Pie Chart"], horizontal=True)
        if view_mode == "Bar Chart":
            st.bar_chart(bias_stats)
        else:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(bias_stats.values(), labels=bias_stats.keys(), autopct="%.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

        st.markdown("### 🏛️ News Source Bias Ratings")
        df_sources = pd.DataFrame(source_bias_data)
        selected_bias_filter = st.radio("Filter by Bias:", ["All", "Left", "Center", "Right"], horizontal=True)
        if selected_bias_filter != "All":
            df_sources = df_sources[df_sources["bias"] == selected_bias_filter]
        st.table(df_sources)

    elif selected_page == "🔎 Custom Search":
        st.title("🔎 Custom Search")

        search_query = st.text_input("🔍 Enter a keyword to search for news:")
        if search_query and (
                len(st.session_state["search_history"]) == 0 or st.session_state["search_history"][-1] != search_query):
            st.session_state["search_history"].append(search_query)

        st.markdown("### 🔄 Search History")
        for query in reversed(st.session_state["search_history"]):
            if st.button(query, key=f"history_{query}"):
                search_query = query
                st.rerun()

        if search_query:
            st.markdown(f"### Showing results for: **{search_query}**")
            search_results = [  ###fake search results
                {"title": "Election Updates", "bias": "Center", "url": "https://news.com/election", "left": 10,
                 "center": 30, "right": 15},
                {"title": "New Economic Policy", "bias": "Right", "url": "https://businessnews.com/policy", "left": 5,
                 "center": 10, "right": 40},
                {"title": "Environmental Crisis", "bias": "Left", "url": "https://greennews.com/crisis", "left": 30,
                 "center": 10, "right": 5},
            ]
            for result in search_results:
                if st.button(f"📌 {result['title']}", key=result['title']):
                    st.session_state["selected_event"] = result
                    st.rerun()

    elif selected_page == "🤖 ML Inference":
        st.title("🤖 Political Stance Inference")

        # Dataset selection
        selected_dataset = st.selectbox(
            "Select Test Dataset:",
            options=list(TEST_DATASETS.keys()),
            index=2
        )

        TEST_DATA_PATH = TEST_DATASETS[selected_dataset]

        # Load model button
        if st.button("🔄 Load Model"):
            with st.spinner("Loading model..."):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model, tokenizer = load_model(BEST_MODEL, MODEL_NAME, device)
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.device = device
                st.success("Model loaded successfully!")

        # Load and display data
        data = load_dataset(TEST_DATA_PATH)

        # Store predictions in session state if not already there
        if 'predictions' not in st.session_state:
            st.session_state.predictions = {}

        # Display data in table
        for idx, item in enumerate(data):
            # Format text with tags
            formatted_text = f"[title] {item['title']} [/title] "
            if 'mention_source' in item:
                formatted_text += f"[source] {item['mention_source']} [/source] "
            formatted_text += f"[content] {item['text']} [/content]"

            # Create expandable section for each item
            with st.expander(f"📄 {item['title']}", expanded=idx in st.session_state.predictions):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write("**Full Text:**")
                    st.write(item['text'])  # Show full text instead of preview

                with col2:
                    # Predict button for each item
                    if st.button("🤖 Predict", key=f"predict_{idx}"):
                        if 'model' not in st.session_state:
                            st.error("Please load the model first!")
                        else:
                            prediction = predict_stance(
                                formatted_text,
                                st.session_state.model,
                                st.session_state.tokenizer,
                                st.session_state.device,
                                labels=LABEL_NAMES
                            )
                            st.session_state.predictions[idx] = prediction

                # Show prediction if available
                if idx in st.session_state.predictions:
                    prediction = st.session_state.predictions[idx]
                    true_label = DATASET_LABEL_NAMES[item['label']]  # Map -2 to 2 range
                    is_correct = prediction['stance'] == true_label

                    st.markdown(
                        f"""
                        <div style="padding: 1rem; border: 1px solid #ddd; border-radius: 0.5rem; margin: 0.5rem 0;">
                            <p><strong>Predicted Stance:</strong> {prediction['stance']}</p>
                            <p><strong>True Stance:</strong> {true_label}</p>
                            <p><strong>Confidence:</strong> {prediction['confidence']:.2%}</p>
                            <p><strong>Match:</strong> {"✅" if is_correct else "❌"}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
