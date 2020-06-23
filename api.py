import flask
from flask import (
    request,
    jsonify
)

import detection_strips
import fluorescence_classifier.analyze as fluore

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return '''<h1>OLASimple IP</h1>
<p>Python service providing the API for image processing of scanned OLASimple diagnostic strips.</p>
<p>Send a POST request to /api/processstrips with the image to be processed.</p>'''


@app.route('/api/processstrips', methods=['POST'])
def process_strips():

    file = request.files['file']

    try:
        tstats = detection_strips.process_image_from_file(file, trimmed=False)
        results = detection_strips.make_calls_from_tstats(tstats)
        response = jsonify(results=results)
    except:
        response = jsonify(
            error='API ERROR: couldnt analyze the file. Be sure file is in a common format like jpg.')

    return response


@app.route('/api/classifyfluorescence', methods=['POST'])
def classify_fluorescence():

    file = request.files['file']
    try:
        results = fluore.glow_box_analysis(file)
        results[1] = [item for sublist in results[1] for item in sublist]
        response = jsonify(results=results)
    except:
        response = jsonify(
            error='API ERROR: couldnt analyze the file. Be sure file is in a common format like jpg.')
    return response


app.run(host='0.0.0.0', port=5000)
