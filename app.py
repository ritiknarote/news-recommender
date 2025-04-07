import streamlit as st
import base64
from fpdf import FPDF
from preprocess import load_dataset
from recommender import NewsRecommender

def generate_pdf(recommendations, df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for idx, score in recommendations:
        headline = df.loc[idx, 'headline']
        description = df.loc[idx, 'description']
        category = df.loc[idx, 'type']

        # Handle NaN values
        if not isinstance(description, str):
            description = "No description available."

        # Safely encode text
        headline = str(headline).encode('latin-1', 'replace').decode('latin-1')
        description = str(description).encode('latin-1', 'replace').decode('latin-1')
        category = str(category).encode('latin-1', 'replace').decode('latin-1')

        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", 'B', size=14)
        pdf.multi_cell(0, 10, headline)

        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, description)
        pdf.cell(0, 10, f"Category: {category}", ln=1)
        pdf.cell(0, 10, f"Similarity Score: {score:.2f}", ln=1)
        pdf.cell(0, 10, "-"*70, ln=1)

    temp_path = "temp_recommendations.pdf"
    pdf.output(temp_path)

    with open(temp_path, "rb") as f:
        pdf_bytes = f.read()
    return pdf_bytes

# Streamlit UI setup
st.set_page_config(page_title="News Recommender", layout="wide")
st.title("🗞️ News Article Recommender")
st.markdown("Get similar news articles using Machine Learning")

# Load dataset
df = load_dataset()
df['text'] = df['headline'] + ' ' + df['content']  # Combine for vectorization

recommender = NewsRecommender(df['text'].tolist())

# Sidebar filters
st.sidebar.header("🔍 Filters")
categories = sorted(df['type'].dropna().unique())
selected_categories = st.sidebar.multiselect("Filter by Category", categories, default=categories)
search_keyword = st.sidebar.text_input("Search by Keyword (optional)")

filtered_df = df[df['type'].isin(selected_categories)]

if search_keyword:
    filtered_df = filtered_df[filtered_df['text'].str.contains(search_keyword, case=False, na=False)]

# Article selection
st.markdown("### Select an article to get recommendations:")
selected_idx = st.selectbox(
    "Choose an article",
    filtered_df.index,
    format_func=lambda x: df.loc[x, "headline"]
)

top_n = st.slider("How many articles to recommend?", min_value=1, max_value=10, value=5)

# Recommendation logic
if st.button("🔁 Recommend Similar Articles"):
    st.subheader("📰 You selected:")
    st.markdown(f"**{df.loc[selected_idx, 'headline']}**")
    st.write(df.loc[selected_idx, 'description'])
    st.write(f"_Category: {df.loc[selected_idx, 'type']}_")
    st.markdown("---")

    st.subheader("📌 Recommended Articles:")
    recommended_indices = recommender.recommend(selected_idx, top_n=top_n)

    for idx, score in recommended_indices:
        if df.loc[idx, 'type'] in selected_categories:
            headline = df.loc[idx, 'headline']
            description = df.loc[idx, 'description']
            link = df.loc[idx, 'news_link']

            if not isinstance(description, str):
                description = "No description available."

            st.markdown(f"### [{headline}]({link})")
            st.write(description)
            st.write(f"📂 Category: {df.loc[idx, 'type']}")
            st.write(f"🔗 **Similarity Score:** {score:.2f}")
            st.markdown("---")

    # PDF Export
    pdf_data = generate_pdf(recommended_indices, df)
    b64 = base64.b64encode(pdf_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="recommendations.pdf">📄 Download as PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
