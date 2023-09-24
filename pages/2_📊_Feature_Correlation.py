import os
import sys
sys.path.append(os.path.dirname(__file__))
from typing import List
import streamlit as st
st.set_page_config(layout='centered')
import pandas as pd
from ModelingTools.Explore.plots import chart_gridplot
from apis import get_corr_matrix, get_chi2_cross

class DataManager:
    @classmethod
    def getData(cls) -> pd.DataFrame:
        return st.session_state['dataframe']
    
    @classmethod
    def getColNames(cls) -> List[str]:
        df = cls.getData()
        return df.columns.tolist()
    
    @classmethod
    def getTypedCols(cls) -> List[str]:
        # only return features that have been labeled types from profiling page
        return list(st.session_state['standalone_config'].keys())
    
    @classmethod
    def getCategCols(cls) -> List[str]:
        cols = []
        for col, conf in st.session_state['standalone_config'].items():
            if conf['dtype'] in ('Binary', 'Nominal', 'Ordinal'):
                cols.append(col)
        return cols
    
    @classmethod
    def getNumCols(cls) -> List[str]:
        cols = []
        for col, conf in st.session_state['standalone_config'].items():
            if conf['dtype'] in ('Numeric'):
                cols.append(col)
        return cols
    
    @classmethod
    def calcCrossCorr(cls):
        # get data
        df = cls.getData()
        
        num_cols = cls.getNumCols()
        categ_cols = cls.getCategCols()
        
        # correlations for numerical vars
        corr_num = get_corr_matrix(df[num_cols])
        # chi2 for categorical vars
        corr_categ = get_chi2_cross(df[categ_cols])
        return corr_num, corr_categ
    
try:
    corr_num, corr_categ = DataManager.calcCrossCorr()
except Exception as e:
    raise e
else:

    st.subheader("Pearson Correlation (Numerical Features)")
    corr_num
    corr_num_chart = chart_gridplot(
        corr_num,
        size = (700, 600),
        title = 'Correlation Coefficients between Numerical Variables'
    )
    st.bokeh_chart(corr_num_chart, use_container_width=True)


    st.subheader("Chi2 P-value (Categorical Features)")
    corr_categ_chart = chart_gridplot(
        corr_categ,
        reverse_color=True,
        size = (700, 600),
        title = 'P value (Chi2 test) between Categorical Variables'    
    )
    st.bokeh_chart(corr_categ_chart, use_container_width=True)