### NLP_text_summarization_apps

THere are two app, nlpapp.py and nlpapp2.py

Both APPs allower the user to input text and have a model or models summarize the text.

* nlpapp.py:  uses a neural net model to read text and generate a summary of the text passed to the app. The App uses a Large BART model from the Huggingface transformer library.The Model uploads once only when app is run and will work of that uploaded model for any text input to the app. 
 

* nlpapp2.py: offers the user the BART model and also the LeX Rank model to summarize text. Lex_rank is much faster and a smaller model than BART. 

Before running the app from the command line there is a requirements.txt file which contains the versions of python libraries used in both apps.

To run either app locally enter in your commandline:  streamlit run nlpapp.py or streamlit run nlpapp2.py

Aug/2021
