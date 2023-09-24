import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from ModelingTools.CustomModel.linear import LinearClfStatsModelWrapper
from ModelingTools.FeatureEngineer.preprocessing import floatFy, get_preliminary_preprocess
from ModelingTools.Explore.profiling import TabularStat
from ModelingTools.FeatureEngineer.preprocessing import get_mutual_info_preprocess
from ModelingTools.FeatureEngineer.feature_selection import chi2_score_cross

def get_anova_num(ts: TabularStat, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    num_cols = ts.get_numeric_cols()
    fs, ps = f_classif(
        X = SimpleImputer(
            strategy='median'
        ).fit_transform(X_train[num_cols]),
        y = y_train
    )
    return pd.DataFrame({
        'Fvalue' : fs, 
        'Pvalue': ps,
        'PvalueLog' : -np.log10(ps)
    }, index = num_cols).sort_values(
        by = 'Fvalue', 
        ascending=False
    )

def get_glm_result(ts: TabularStat, X_train: pd.DataFrame, y_train: pd.Series) -> str:
    ftarget_preprocess_pipe = get_preliminary_preprocess(ts)
    lm = LinearClfStatsModelWrapper(
        model_family='logit',
        fit_intercept=True,
    )
    lm_pipe = Pipeline([
        ('preprocess', ftarget_preprocess_pipe),
        ('float', FunctionTransformer(floatFy)),
        ('glm', lm)
    ])
    # train
    lm_pipe.fit(X_train, y_train)
    return lm_pipe['glm'].model_result_.summary()

def get_mutual_info(ts: TabularStat, X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
    ftarget_preprocess_pipe = get_mutual_info_preprocess(ts)  # all discrete features will be ordinal encoded
    X_processed = ftarget_preprocess_pipe.fit_transform(X_train, y_train)

    mi = mutual_info_classif(
        X_processed, 
        y_train, 
        discrete_features = X_processed.columns.isin(ts.get_categ_cols()),
        n_neighbors = 3
    )
    mi = pd.Series(
        mi, 
        index = X_processed.columns
    ).sort_values(ascending = False)
    return mi

def get_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    grid = df.corr()
    grid.index.name = 'Feature'
    grid.columns.name = 'Features'
    return grid

def get_chi2_cross(df: pd.DataFrame) -> pd.DataFrame:
    chi_test_output = chi2_score_cross(df)
    # convert to grid
    grid = chi_test_output.pivot(
        index='var1', 
        columns='var2', 
        values='pvalue'
    )
    grid.index.name = 'Feature'
    grid.columns.name = 'Features'
    return grid