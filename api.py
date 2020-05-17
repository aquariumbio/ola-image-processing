import flask
from flask import request, jsonify, Response

import json
import olaip
import inspect

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return '''<h1>OLASimple IP</h1>
<p>Python service providing the API for image processing of scanned OLASimple diagnostic strips.</p>
<p>Send a POST request to /api/processimage with the image to be processed.</p>'''


@app.route('/api/processimage', methods=['POST'])
def process():
    
    file = request.files['file']

    try:
        tstats = olaip.process_image_from_file(file, trimmed=False)    
        results = olaip.make_calls_from_tstats(tstats)
        response = jsonify(results=results)
    except:
        response = jsonify(error='API ERROR: couldnt analyze the file. Be sure file is in a common format like jpg.')
    
    return response


app.run(host='0.0.0.0', port=5000)
