import numpy as np
import pandas as pd
import streamlit as st
from streamlit import caching 
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig

#App works off large BART mdel. Model  is downloaded once the app runs and is cached. There are quicker models to summarize using generative models but results are good with ths model.

# Loading the model and tokenizer for bart-large-cnn
@st.cache(allow_output_mutation=True)
def get_model():
	tokenizer=BartTokenizer.from_pretrained('facebook/bart-large-cnn')
	model=BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
	return tokenizer,model
#oonce run runs model and tokenizer are cached
tokenizer,model= get_model() 


#use pretrained model and toekizer to produce summary tokens and then decode thise summarized tokens 
@st.cache(allow_output_mutation=True)
def summarizer(original_text):
	inputs = tokenizer.batch_encode_plus([original_text],return_tensors='pt')
	summary_ids = model.generate(inputs['input_ids'], early_stopping=True)
	bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
	return bart_summary

 


def main():
	""" NLP Based App with Streamlit """

	# Title
	st.title("Summarize text")

	# Summarization
	

	message = st.text_area("Enter Text")	 
	if st.button("Summarize"):
			 
			summary_result = summarizer(message)
			#st.write(summary_result)

	
			st.success(summary_result)


	 
	

if __name__ == '__main__':
	main()