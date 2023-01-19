# ==============================================================================
# title              : app.py
# description        : This is the flask app for Bert closed domain chatbot which accepts the user request and response back with the answer
# author             : Pragnakalp Techlabs
# email              : letstalk@pragnakalp.com
# website            : https://www.pragnakalp.com
# python_version     : 3.6.x +
# ==============================================================================

# Import required libraries
from flask import Flask, render_template, request
from flask_cors import CORS
import email
import csv
import datetime
import smtplib
import ssl
import socket
from email.mime.text import MIMEText
from bert import QA

timestamp = datetime.datetime.now()
date = timestamp.strftime('%d-%m-%Y')
time = timestamp.strftime('%I:%M:%S')
IP = ''

app = Flask(__name__)
CORS(app)

# Provide the fine_tuned model path in QA Class
model_tr = QA("turkish_model_bin")

# This is used to show the home page
@app.route("/")
def home():
    return render_template("home.html")

# This is used to give response 
@app.route("/predict")
def get_bot_response():   
    IP = request.remote_addr
    q = request.args.get('msg')
    bert_bot_log = []
    bert_bot_log.append(q)
    bert_bot_log.append(date)
    bert_bot_log.append(time)
    bert_bot_log.append(IP)
    
    # You can provide your own paragraph from here
    turkish_para = "Google, 1998 yılında Larry Page ve Sergey Brin tarafından Ph.D. Kaliforniya'daki Stanford Üniversitesi öğrencileri. Birlikte hisselerinin yaklaşık yüzde 14'üne sahipler ve hisse senedi oylama gücünün yüzde 56'sını hisse senetlerini denetleyerek kontrol ediyorlar. Google'ı 4 Eylül 1998'de özel bir şirket olarak dahil ettiler. İlk halka arz (IPO) 19 Ağustos 2004'te gerçekleşti ve Google, Googleplex lakaplı California'daki Mountain View'daki genel merkezine taşındı. Ağustos 2015'te Google, çeşitli ilgi alanlarını Alphabet Inc adlı bir holding olarak yeniden düzenlemeyi planladığını açıkladı. Google, Alfabenin önde gelen yan kuruluşudur ve Alfabenin İnternet çıkarları için şemsiye şirket olmaya devam edecektir. Sundar Pichai, Alfabenin CEO'su olan Larry Page'ın yerine Google'ın CEO'su olarak atandı."

    # This function creates a log file which contain the question, answer, date, time, IP addr of the user
    def bert_log_fn(answer_err):
        bert_bot_log.append(answer_err)
        with open('bert_bot_log.csv', 'a' , encoding='utf-8') as logs:
            write = csv.writer(logs)
            write.writerow(bert_bot_log)
        logs.close()

    # This block calls the prediction function and return the response
    try:        
        out = model_tr.predict(turkish_para, q)
        confidence = out["confidence"]
        confidence_score = round(confidence*100)
        if confidence_score > 30:
            bert_log_fn(out["answer"])
            return out["answer"]
        else:
            bert_log_fn("Sorry I don't know the answer, please try some different question.")
            return "Sorry I don't know the answer, please try some different question."         
    except Exception as e:
        bert_log_fn("Sorry, Server doesn't respond..!!")
        print("Exception Message ==> ",e)
        return "Sorry, Server doesn't respond..!!"

# You can change the Flask app port number from here.
if __name__ == "__main__":
    app.run(port='3000')
