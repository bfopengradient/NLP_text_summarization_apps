import numpy as np
import pandas as pd
import streamlit as st
from streamlit import caching 
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

#App works off two NLP models. The large BART model is downloaded once the app runs and is cached. Caching will enable you to import BART once and 
#test the trained BART model on as much tesxt as is needed without having to upload the model ooevr and over. BTW The Lex_Rank model is much smaller and faster.

# Loading the model and tokenizer for bart-large-cnn
@st.cache(allow_output_mutation=True)
def get_model():
	tokenizer=BartTokenizer.from_pretrained('facebook/bart-large-cnn')
	model=BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
	return tokenizer,model
#oonce run runs model and tokenizer are cached
tokenizer,model= get_model() 


#Use pretrained model and tonekizer to produce summary tokens and then decode thise summarized tokens 
@st.cache(allow_output_mutation=True)
def summarizer(original_text):
	inputs = tokenizer.batch_encode_plus([original_text],return_tensors='pt')
	summary_ids = model.generate(inputs['input_ids'], early_stopping=True)
	bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
	#return BART summary
	return bart_summary

#Load in and use Lex_rank in one function.  Model is much smaller and faster
@st.cache(allow_output_mutation=True)
def sumy_summarizer(original_text):
	# Initializing the parser
	my_parser = PlaintextParser.from_string(original_text,Tokenizer('english'))
	# Creating a summary of 3 sentences.
	lex_rank_summarizer = LexRankSummarizer()
	lexrank_summary = lex_rank_summarizer(my_parser.document,sentences_count=3)
	# return the summary 
	for sentence in lexrank_summary:
	  return sentence


def main():
	""" NLP Based App with Streamlit. Choice of two mdodels. BART Large or Sumy Lex Rank """

	# Title
	st.title("Summarize text with NLP models")
	st.write("#")

	 
	# Summarization	
	message = st.text_area("Enter Text in box below")
	st.write('#')

	#provide choice of model. Bart is much slower than Lex_rank
	col1, col2 = st.beta_columns(2)
	with col1:
		bart_button=st.button('Summarzie with BART model')
		if bart_button:
			summary_result = summarizer(message)
			st.success(summary_result)
	with col2:
		sumy_button=st.button('Summarize with Lex_rank model')
		if sumy_button:
			summary_result = sumy_summarizer(message)
			st.success(summary_result)

if __name__ == '__main__':
	main()