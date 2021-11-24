# Healthcare Startup ML


## Overview

1. Uploaded data to Google Drive and evaluated in Google Colab
2. Trained data using `trainer.py`
3. Modified `api.py` to provide three endpoints:
   1. http://localhost:8080/ : base URL for sanity checking
   2. http://localhost:8080/predict : API endpoint accepting json POST data
   3. http://localhost:8080/upload : webform allowing CSV (or zipped CSV), or json data
4. Wrapped model/api in a Dockerfile for consistency


## Model Changes
1. Subclassed ColumnTransformer to create a custom transform
   1. Added two new features to indicate groups with "large" value_counts.
2. Used xgboost with a light grid search


## Required API Files
1. `api.py` : flask API python module
2. `saved_model.joblib` : serialized model object
3. `Dockerfile` : Docker configuration file
4. `requirements.txt` : required libraries in Docker image


## Running the model
1. Convert training zip to json for use with final API. Code available in `utils.py`.

    ```python
    import pandas as pd
    df = pd.read_csv('train.zip')
    df.to_json('train.json', orient='index')
    ```
2. Clone the repo, and build / run the Docker container.
    
    ```shell
    git clone git@github.com:sysgenerated/healthcare-startup.git
    docker build -t healthcare-startup .
    docker run -t -p 8080:8080 healthcare-startup
    ```
3. POST json data to the API running on port 8080.

    ```shell
    curl -X POST -H "Content-Type: application/json; charset=utf-8" -d @train.json "http://localhost:8080/predict"
    ```
4. Batch predictions can also be performed by uploading a CSV (or zipped CSV), or json file to a webform.
http://localhost:8080/upload


## Example model input
===============

train.json
----

```json
{
  "0": {
    "claim_amount": 1040.13,
    "claim_id": "edec6a2f-f49e-4db3-8828-2e7f44d48864",
    "drg": 24,
    "is_medicaid": false,
    "is_medicare": true,
    "npi": 8000148798,
    "paid_amount": 676.08,
    "patient_age": 51,
    "payer_name": "zxiVJcrvbPtJtXlX"
  },
  "1": {
    "claim_amount": 1165.98,
    "claim_id": "3992ef78-1fcb-4423-8d9e-a0fbeb753053",
    "drg": 64,
    "is_medicaid": false,
    "is_medicare": true,
    "npi": 1557768496,
    "paid_amount": 749.6,
    "patient_age": 41,
    "payer_name": "ZqHRFBrwwWIvGXIp"
  }
}
```


## Example model output
===============

predictions.json
----

```json
{
  "predictions": {
    "edec6a2f-f49e-4db3-8828-2e7f44d48864": 679.77313,
    "3992ef78-1fcb-4423-8d9e-a0fbeb753053": 754.2513
  }
}
```


## License

Distributed under the MIT License. See `LICENSE` for more information.
