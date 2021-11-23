import pandas as pd
import joblib

from flask import Flask, Response, request

app = Flask(__name__)
model = joblib.load('saved_model.joblib')


# Utility function to return file extension
def file_extension(filename):
    return filename.rsplit('.', 1)[1].lower()


# HTML for file upload route
file_upload_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Claim Predictor</title>
</head>
<body>
    <h1>File Upload</h1>
    <p>This form accepts csv (or zipped csv) or json files</p>
    <form method="POST" action="" enctype="multipart/form-data">
      <p><input type="file" name="file", accept=".csv,.json,.zip"></p>
      <p><input type="submit" value="Submit"></p>
    </form>

</body>
</html>
"""


# Base route to confirm flask accepts connections
@app.route('/')
def index():
    return 'Use /predict as the api endpoint or /upload for file upload predictions!'


# Main API endpoint accepting only json files
# Ex: curl -X POST -H "Content-Type: application/json; charset=utf-8" -d @train.json "http://localhost:8080/predict"
@app.route('/predict', methods=['POST'])
def predict():
    payload = request.json
    df = pd.DataFrame.from_dict(payload, orient='index').set_index('claim_id')
    predictions = model.predict(df)
    return {'predictions': dict(zip(df.index, predictions))}


# Upload form for batch predictions
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        payload = request.files['file']
        extension = file_extension(payload.filename)
        if extension == 'csv':
            df = pd.read_csv(payload).set_index('claim_id')
        elif extension == 'zip':
            df = pd.read_csv(payload, compression='zip').set_index('claim_id')
        elif extension == 'json':
            df = pd.read_json(payload, orient='index').set_index('claim_id')
        predictions = model.predict(df)
        content = {'predictions': dict(zip(df.index, predictions))}
        content = str(content).replace("'", "\"")
        return Response(content,
                        mimetype='application/json',
                        headers={'Content-Disposition': 'attachment;filename=predictions.json'})
    else:
        return file_upload_html


# Note: Changed port from 81 to 8080 so linux doesn't complain
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
