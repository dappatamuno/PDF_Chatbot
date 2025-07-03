import streamlit as st
from extract import extract_text_from_pdf
from qa import PDFChatbot, split_text

st.title("📄 Chat with Your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success("✅ PDF Uploaded Successfully!")

    with st.spinner("Extracting and indexing..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(text)
        chatbot = PDFChatbot()
        chatbot.build_index(chunks)

    st.success("✅ Ready! Ask your question below:")
    query = st.text_input("Ask a question about the document:")

    if query:
        answers = chatbot.retrieve_answer(query)
        st.markdown("### 🔍 Relevant Passages:")
        for i, ans in enumerate(answers):
            st.markdown(f"**{i+1}.** {ans}")
