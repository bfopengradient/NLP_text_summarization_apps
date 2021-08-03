### NLP_text_summarization_apps

There are two apps: nlpapp.py and nlpapp2.py

Both APPs allower the user to input text and have a model or models summarize the text.

* nlpapp.py:  uses a neural net model to read text and generate a summary of the text passed to the app. The App uses a Large BART model from the Huggingface transformer library.The Model uploads once only when app is first run. After uploading you can test the pretrained model on text as much as is needed. 
 

* nlpapp2.py: offers the user the BART model and also the LexRank model to summarize text. LexRank is much faster and a smaller model than BART. 

Before running the app from the command line there is a requirements.txt file which contains the python libraries used in both apps. pip install the requirements.txt if needed ahead of running the apps.

To run either app locally enter in your commandline:  streamlit run nlpapp.py or streamlit run nlpapp2.py


BART research paper: https://arxiv.org/abs/1910.13461

LexRank research paper: https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html

Aug/2021
