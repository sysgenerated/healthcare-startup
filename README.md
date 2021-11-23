# Data Science Candidate Repo
For this challenge, you are given a CSV dataset (compressed as a.zip) and a pre-trained linear regression model (model.joblib). This dataset and your objectives reflect the kinds of problems we solve here at Janus.

**No actual patient information is included in this dataset.**

## Objectives
Your objectives are:
- Write a Python script that trains a model to predict `paid_amount`
- Update the provided Flask API script to serve predictions made by your model
- Write either a Python script or Jupyter notebook that includes your technical analyses and findings

## Dataset

The CSV dataset (train.zip) consisting of the following columns:

| column | dtype | desc | Target? |
|---|---|---|---|
| `claim_amount` | FLOAT | Amount billed by the healthcare provider for a patient visit  | No |
| `claim_id` | STRING  | Unique alpha-numeric identifier for a given claim | No |
| `drg` | STRING | Diagnosis Related Group code for patient visit  | No |
| `npi` | STRING | National Provider Identifier code for healthcare provider  | No |
| `is_medicaid` | BOOLEAN |  Flag indicating whether this is a medicaid charge | No |
| `is_medicare` | BOOLEAN |  Flag indicating whether this is a medicare charge | No |
| `paid_amount` | FLOAT | The final amount paid by the insurance provider. | Yes |
| `payer_name` | STRING |  Name of payer (i.e. insurance provider) | No |
| `patient_age` | INTEGER | A patient's age in years  | No |


## Evaluation
We will evaluate your work sample by using your prediction script to make predictions
against a test set that is hidden from you.
Please be sure to give careful instructions explaining how to run your code
to make these predictions. 

You will be evaluated via the following criteria:
- Readability and cleanliness of your code
- Ease of running your code to make new predictions
- Quality of analysis
- Improvement over provided model



## My Notes
docker build -t healthcare-startup .
docker run -t -p 8080:8080 healthcare-startup