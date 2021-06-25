"""

@author: Akshay
"""


from flask import Flask, render_template,request
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app=Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def detect():
    try:
        model=joblib.load('spam_model.pkl') # Loading our model
        message= request.form['message']  # Input message from the home page
        ''' Inputs the messages in the data
        1. Removes stopwords
        2. Removes punctuations or any other special character'''
        
        stop_words=set(stopwords.words('english'))
        words=word_tokenize(message.lower())
        filtered_msg=[]
        for word in words:
            if(word not in stop_words and word.isalnum()):
                filtered_msg.append(word)
        msg= [' '.join(filtered_msg)]
        
        vectorizer=joblib.load('count_vectorizer.pkl')  # vectorizer to convert the input message to sparse vector
        msg_vector=vectorizer.transform(msg)
        
        # Predict whether its a spam or not
        if(model.predict(msg_vector)==1):
            result="It is a spam message. Please ignore it."
        else:
            result="Important message"
        if(message.lstrip()==''):
            return home()
        print(message)
        return render_template('index.html', message=message, prediction=result)
    except:
        return "Something is wrong"

if __name__ == "main":
    app.run(use_reloader=False)