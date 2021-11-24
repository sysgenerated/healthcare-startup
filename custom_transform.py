from sklearn.base import BaseEstimator, TransformerMixin


# Create a customer Column Transformer to identify "large" the large drg, nri group
class GroupCountCliff(BaseEstimator, TransformerMixin):
    def __init__(self):
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
