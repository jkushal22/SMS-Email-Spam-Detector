import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import nltk

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

# nltk.corpus has a library stopwords which contains words like 
# me their what etc which dont hold much value in the meaning of the sentence rather used for its formation

    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    
    return " ".join(y)

model = pickle.load(open('/Users/kushaljirawla/Desktop/Untitled Folder/model.pkl','rb'))
tf = pickle.load(open('/Users/kushaljirawla/Desktop/Untitled Folder/vectorizer.pkl','rb'))

st.title('Email/SMS Classifier')

input_sms = st.text_input('Enter a message')

if st.button('Predict'):

    # preprocessing
    transform_sms = transform_text(input_sms)
    
    #vectorizing
    vector_input = tf.transform([transform_sms])
    
    #prediction
    result = model.predict(vector_input)[0]
    
    #output
    if result == 1:
        st.text("SPAM")
        
    else:
        st.text('NOT SPAM')
