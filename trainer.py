import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from joblib import dump, load


# Set booleans to control training logic
grid_search = True
save_model = False


# Load provided baseline model and create a new xgb model
baseline_model = load('baseline_model.joblib')
model = XGBRegressor(random_state=2021, n_estimators=900, max_depth=5, learning_rate=0.2, subsample=1.0)


# Create gridsearch parameter space for xgb
param_grid = {
    'regressor__n_estimators': [800, 900, 1000],
    'regressor__learning_rate': [0.1, 0.2, 0.3],
    'regressor__max_depth': [3, 5, 7],
    'regressor__subsample': [0.9, 1.0]}


# Load data with known index and target columns
def load_data(data):
    X = pd.read_csv(data).set_index('claim_id')
    y = X.pop('paid_amount')
    return X, y


# Train a model and return the model object
def generate_finalized_model(model, X_train, y_train):
    # Remove outliers where pct paid is very low
    paid_ratio = y_train / X_train['claim_amount']
    keep_rows = paid_ratio > 0.01
    y_train = y_train[keep_rows]
    X_train = X_train[keep_rows]
    # Train the model
    model.fit(X_train, y_train)
    return model


# Create a customer Column Transformer to identify "large" the large drg, nri group
class GroupCountCliff(BaseEstimator, TransformerMixin):
    def __init__(self, cliffs=2):
        self.cliffs = cliffs
        self.col_set = {}

    # This method looks for a large change in value_counts and uses that point
    # as a split point to identify "large" value_counts
    def generate_cliff_map(self, df):
        counts = df.value_counts().sort_values(ascending=False)
        counts = counts.to_frame()
        col_name = counts.columns[0]
        cliff_col_name = col_name + '_cliff'
        counts[cliff_col_name] = counts[col_name].shift(1)
        counts[cliff_col_name] = counts[col_name] / counts[cliff_col_name]
        counts = counts.reset_index()
        counts_cliff = counts.index[counts[cliff_col_name] == counts[cliff_col_name].min()].to_list()[0]
        indicator_rows = counts.iloc[0:counts_cliff]['index'].values
        return indicator_rows

    def fit(self, X, y=None):
        for col in X.columns:
            self.col_set[col] = self.generate_cliff_map(X[col])
        return self

    def transform(self, X, y=None):
        for col in X.columns:
            cliff_col_name = col + '_cliff'
            X[cliff_col_name] = 0
            X.loc[X[col].isin(self.col_set[col]), cliff_col_name] = 1
        return X


# Load train / test data
X,y = load_data("./train.zip")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Identify columns to be transformed
cliff_cols = ['drg', 'npi']
onehot_cols = ['drg', 'npi', 'payer_name', 'is_medicaid', 'is_medicare']
scale_cols = ['claim_amount', 'patient_age']


# Create an sklearn Column Transformer object
preprocessor = ColumnTransformer(transformers=[('cliff', GroupCountCliff(), cliff_cols),
                                               ('onehot', OneHotEncoder(), onehot_cols),
                                               ('scaler', StandardScaler(), scale_cols)])

# Create an sklearn model Pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', model)])


# Configure grid search if boolean is set
if grid_search:
    model = GridSearchCV(model, param_grid, n_jobs=4)


# Train the model
finalized_model = generate_finalized_model(model, X_train, y_train)


# Print grid search results if enabled
if grid_search:
    print(model.best_params_)


# Retrain using all available data and save model to serialized object
if save_model:
    finalized_model = generate_finalized_model(model, X, y)
    dump(finalized_model, "saved_model.joblib", compress=True)


# Scratch work for model results
results = pd.DataFrame({'actual': y_test,
                        'baseline_predicted': baseline_model.predict(X_test),
                        'retrain_predicted': model.predict(X_test)})

baseline_r_squared = r2_score(results['actual'], results['baseline_predicted'])
retrain_r_squared = r2_score(results['actual'], results['retrain_predicted'])
