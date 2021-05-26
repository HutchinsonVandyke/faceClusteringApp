import os
import glob
from face_recognition import processOneImage
from os.path import join, dirname, realpath
from HOG_kmeans import predict
from dataLoader import createRepDicts
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from pathlib import Path

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)

""" model = load_model('deployment_28042020')
cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region'] """

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/upload',methods=['POST'])
def upload():
    
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
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('show_upload',
                                    filename=filename))
    return render_template("home.html")

@app.route('/upload/<filename>')
def show_upload(filename):
    
    root = os.getcwd()

    dirpath = os.path.join(root, "static", "uploads", filename)
    hogPath = os.path.join("." , "static", "uploads", filename)
    hog = processOneImage(hogPath)
    #got to load pickle reps so can predict
    ageReps = createRepDicts([], "age")
    ethnicityReps = createRepDicts([], "ethnicity")
    genderReps = createRepDicts([], "gender")
    agePrediction = predict(ageReps, hog)
    ethnicityPrediction = predict(ethnicityReps, hog)
    genderPrediction = predict(genderReps, hog)
    
    if ethnicityPrediction == 0:
        ethnicityPrediction = "caucasian"   
    elif ethnicityPrediction == 1:
        ethnicityPrediction = "african"
    elif ethnicityPrediction == 2:
        ethnicityPrediction = "asian"
    elif ethnicityPrediction == 3:
        ethnicityPrediction = "indian"
    else:
        ethnicityPrediction = "hispanic"

    if genderPrediction == 0:
        genderPrediction = "male"
    else:
        genderPrediction = "female"
   
    files = glob.glob('./static/uploads/*')
    for f in files:
        os.remove(f)

    return render_template("prediction.html", _age = agePrediction, _gender = genderPrediction, _ethnicity = ethnicityPrediction)

if __name__ == '__main__':
    app.run(debug=True)