import validators, streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader 

# streamlit app
st.set_page_config(page_title="Buat Ringkasan dari Youtube atau Website")
st.title("Buat Ringkasan dari Youtube atau Website")
st.subheader("URL Ringkasan")

api_key = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(
    groq_api_key=api_key, 
    model_name="llama-3.3-70b-versatile",
    streaming=True,
    temperature=0.3
)


generic_url=st.text_input("URL", label_visibility="collapsed")
language_input=st.selectbox("Pilih Bahasa",
                            ("Indonesia", "English"))

# Prompt template
prompt_template = """
You are a professional assistant that summarizes content accurately.
Please provide a comprehensive summary of the text provided below.

The entire summary MUST be written in {language} language.
Format the output as follows:
1. A brief executive summary (2-3 sentences).
2. Key points in a bulleted list.

TEXT TO SUMMARIZE:
"{text}"

SUMMARY IN {language}:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text", "language"])

if st.button("Buat Ringkasan"):
    ## validate all the input
    if not generic_url.strip():
        st.error("Masukkan informasi untuk memulai")
    elif not language_input:
        st.error("Pilih Bahasa")
    elif not validators.url(generic_url):
        st.error("Masukkan url yang valid seperti video youtube atau website")
    else:
        try:
            with st.spinner("sedang meringkas..."):
                # loading the website or youtube data
                if 'youtube.com' in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, 
                                                            add_video_info=False, 
                                                            language=["id", "en"])
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False, 
                                                   headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"})
                docs = loader.load()

                # Chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run({
                    "input_documents": docs, 
                    "language": language_input})

                st.success(output_summary)

        except Exception as e:
            st.exception(e)

    

