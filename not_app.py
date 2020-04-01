import flask
import os
from flask import jsonify, request , render_template
from flask import flash, redirect, url_for, session
from joblib import load
import requests, json
import pandas as pd
import requests
import random
import subprocess
import glob
from random import random
import re
from twilio.rest import Client
import torch
import os
from flask_ngrok import run_with_ngrok
import numpy as np
import sklearn
from joblib import load as ld
import requests
import pyimgur
import time



PREDMOD = ld('rf.joblib')
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}

account_sid = '*********'
auth_token = '********'

client = Client(account_sid, auth_token)


app = flask.Flask(__name__ , 
            static_url_path='', 
            static_folder='static')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["DEBUG"] = True
app.secret_key = 'super secret key'


#Global Variables
RESPONSE = {}
CONTACT = { "police" : '+917756915727' , "ambulance" : '+918655513317' , "fire_brigade" : '+919762622540' }
# CONTACT = { "police" : '+919762622540' , "ambulance" : '+918655513317' , "fire_brigade" : '+919762622540' }
NUMBERS = [ '+919762622540' , '+918655513317' ]
CONTENT = { "police" : 'Need Police Enforcement at ' , "ambulance" : 'Need Ambulance at ' , "fire_brigade" : 'Dispatch Fire Brigade' }
LINKS = {}
HOME_URL = [None]



UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['DEBUG'] = True
app.secret_key = 'DEAD BOY'

run_with_ngrok(app)   #starts ngrok when the app is run

CLIENT_ID = "*********"
PATH = "**********"
IMC = pyimgur.Imgur(CLIENT_ID)

@app.route("/test")
def home():
    return "<h1>Running Flask on Google Colab! with " + torch.cuda.get_device_name(0) + "</h1>" 
  
def get_ngrok_url():

  if HOME_URL[0] is None:    
    localhost_url = "http://localhost:4040/api/tunnels"  # Url with tunnel details
    time.sleep(1)
    tunnel_url = requests.get(localhost_url).text  # Get the tunnel information
    j = json.loads(tunnel_url)

    tunnel_url = j['tunnels'][0]['public_url']  # Do the parsing of the get
    tunnel_url = tunnel_url.replace("https", "http")
    HOME_URL[0] = tunnel_url

  return HOME_URL[0]

def get_random_addr():

  i = int( random()*84 ) % 4
  addr = [ ' Ramwadi , Virar ', ' NL Complex, Dahisar ' , ' Khaugaon , Pune ' , ' Bandstand , Bandra ']

  return addr[i]


def sos(typ):
  addr = get_random_addr()
  content = CONTENT[typ]
  msg = content + addr + ' ( this is a College Project test msg , dont respond ) '
  message = client.messages \
                  .create(
                      body=msg,
                      from_='+12015849601',
                      to=CONTACT[typ]
                  )

  print("message" , message.status , "to" , message.to)


@app.route('/respond/<objId>', methods=['GET', 'POST'])
def respond(objId):
  data = list(request.form.keys())

  for d in data:
      RESPONSE[objId][d] += 1

      if RESPONSE[objId][d] > 1:
          sos(d)

  print(objId)

  return "Thank You For Your Response"

def secure_filename(name):
  return "test.mp4"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_images(images):
  links = []

  for im in images:
    uploaded_image = IMC.upload_image(im, title="Test Image for BE Project" , album='Sfc4ZTfUkC9DG5H')
    links.append( uploaded_image.link )
  
  return links


def send_form_sms(url):

  msg = " Need help in identifying the Responder , Click on url provide assistance " \
        + url + " ( this is a College Project test msg  ) "
  

  for contact in NUMBERS:
    message = client.messages \
                    .create(
                        body=msg,
                        from_='+12015849601',
                        to=contact
                    )


    print("message" , message.status , "to" , message.to)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("upload successful")
            return redirect( url_for('result') )
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/result', methods=['GET'])
def result():

  mfe.predicter()

  subprocess.call("rm -r static/*" , shell = True)
  

  for f in glob.glob('features/*.npy'):
    fpath = f
    data = np.load(fpath)[0]
    print(data[1:10])
    result = PREDMOD.predict([data])
    if result == 0:
      objId = str(int(random()*65717))
      subprocess.call("mkdir static/"+ objId , shell=True)

      RESPONSE[objId] = {"police": 0 , "ambulance" : 0 , "fire_brigade" : 0}

      frame_command = "ffmpeg -i uploads/test.mp4 -vf thumbnail=120,setpts=N/TB -r 1 -vframes 30 static/"+objId+"/if%03d.png"

      subprocess.call(frame_command , shell=True)

      images_ = glob.glob('static/'+objId+'/*.png')
      LINKS[objId] = upload_images(images_)
      help_url = get_ngrok_url() + '/help/' + objId
      send_form_sms(help_url)
      return "Abnormal Video"
       
    else:
      return "Normal Video"
  
  return "error"

@app.route('/help/<objId>', methods=['GET'])
def help(objId):
    
    return render_template('check.html' , images=LINKS[objId] , objId=objId)


app.run()