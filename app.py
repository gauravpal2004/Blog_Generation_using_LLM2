import streamlit as st
from transformers import pipeline

## Function To get response from LLAma 2 model

def getLLamaresponse(input_text, no_words, blog_style):

    # Load LLama 2 model pipeline
    llm_pipeline = pipeline("text-generation", model="llama-2-7b-chat.ggmlv3.q8_0.bin")

    ## Prompt Template
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
            """

    prompt = template.format(blog_style=blog_style, input_text=input_text, no_words=no_words)

    ## Generate the response from the LLama 2 model
    response = llm_pipeline(prompt, max_length=512, do_sample=True, temperature=0.01)[0]['generated_text']
    print(response)
    return response


st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

## creating to more columns for additional 2 fields

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for',
                              ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))
