import os
import sys
sys.path.append(os.path.dirname(__file__))
import streamlit as st
st.set_page_config(layout='centered')
import pandas as pd
from typing import List, Tuple, Dict, Union, Literal
from bokeh.models import Panel, Tabs
from ModelingTools.Explore.profiling import _BaseUniVarClfTargetCorr, CategUniVarClfTargetCorr, NumericUniVarClfTargetCorr,\
        TabularUniVarClfTargetCorr
from ModelingTools.Explore.plots import chart_barchart, chart_segment_group_count, chart_segment_group_perc, chart_boxplot, chart_optbin_multiclass
from apis import get_glm_result, get_anova_num, get_mutual_info

class DataManager:
    @classmethod
    def getData(cls) -> pd.DataFrame:
        return st.session_state['dataframe']
    
    @classmethod
    def getColNames(cls) -> List[str]:
        df = cls.getData()
        return df.columns.tolist()
    
    @classmethod
    def rememberSelection(cls, targetname:str, colname: str):
        st.session_state['selection'] = {
            'target' : targetname,
            'feature' : colname
        }
        
    @classmethod
    def getTypedCols(cls) -> List[str]:
        # only return features that have been labeled types from profiling page
        return list(st.session_state['standalone_config'].keys())
        
    @classmethod
    def getSelections(cls) -> dict:
        if 'selection' in st.session_state:
            return st.session_state['selection']
        else:
            # return inital value
            cols = cls.getColNames()
            targetname = cols[0]
            feature_cols = [col for col in cls.getTypedCols() if col != targetname]
            return {
                'target' : targetname,
                'feature' : feature_cols[0]
            }
            
    @classmethod
    def calcUniCorr(cls):
        configs = {}
        for col, conf in st.session_state['standalone_config'].items():
            if conf['dtype'] in ('Binary', 'Nominal', 'Ordinal'):
                configs[col] = CategUniVarClfTargetCorr()
            elif conf['dtype'] in ('Numeric'):
                configs[col] = NumericUniVarClfTargetCorr()
        
        # get data
        ycol = cls.getSelections()['target']
        df = cls.getData()
        X_train = df[cls.getTypedCols()].drop(columns = [ycol], errors='ignore')
        y_train = df[ycol]
        
        progress_bar = st.progress(0, text='Running Univaraite Correlation Charts')
        # train
        tuvct = TabularUniVarClfTargetCorr(configs = configs)
        tuvct.fit(X_train, y_train)
        # save
        st.session_state['corr'] = tuvct
        
        # calculate metrics
        if 'tabular_stat' in st.session_state:
            ts = st.session_state['tabular_stat']['obj']
        else:
            st.error("Tabular Stat does not exist, go back to profiling page and click [Train Dataset]")

        progress_bar.progress(50, text = 'Running Mutual Info Stat')
        try:
            mi = get_mutual_info(ts, X_train=X_train, y_train=y_train)
        except Exception as e:
            if 'mutual_info' in st.session_state:
                del st.session_state['mutual_info']
            st.session_state['mutual_info_err'] = str(e)
        else:
            st.session_state['mutual_info'] = mi
        
        progress_bar.progress(75, text = 'Running GLM test')
        try:
            glm_str = get_glm_result(ts, X_train=X_train, y_train=y_train)
        except Exception as e:
            if 'glm_str' in st.session_state:
                del st.session_state['glm_str']
            st.session_state['glm_err'] = str(e)
        else:
            st.session_state['glm_str'] = glm_str
        
        progress_bar.progress(90, text = 'Running ANOVA test')
        try:
            anova_p = get_anova_num(ts, X_train=X_train, y_train=y_train)
        except Exception as e:
            if 'anova_p' in st.session_state:
                del st.session_state['anova_p']
            st.session_state['anova_err'] = str(e)
        else:
            st.session_state['anova_p'] = anova_p['PvalueLog']
        
        progress_bar.progress(100, text = 'Complete!')
        
        
        
    @classmethod
    def getUniCorrPlot(cls, featurename: str) -> dict:
        if 'corr' not in st.session_state:
            raise KeyError("Univariate Correlation not yet calculated, click button to calculate")
        if featurename in st.session_state['standalone_config']:
            config = st.session_state['standalone_config'][featurename]
            tuvct_configs = st.session_state['corr'].configs
            if featurename in tuvct_configs:
                figs = cls.getProfileFigs(
                    dtype = config['dtype'],
                    stat = tuvct_configs[featurename]
                )
                return figs
            else:
                raise KeyError("Feature newly added, please recalculate")
        else:
            raise KeyError("feature Dtype not yet determined, please go back to profiling page and add it")
        
    @classmethod
    def getProfileFigs(cls, dtype:str, stat: _BaseUniVarClfTargetCorr) -> dict:
        if dtype in ('Binary', 'Nominal', 'Ordinal'):
            tabs_x_y = Tabs(
                tabs = [
                    Panel(
                        child = chart_segment_group_count(
                            stat.p_x_y_,
                            group_name=stat.yname_, 
                            agg_name=stat.colname_,
                            size=(600, 500),
                            title = "P(x|y) - Category Count By Target"
                        ),
                        title="Count"
                    ),
                    Panel(
                        child = chart_segment_group_perc(
                            stat.p_x_y_,
                            group_name=stat.yname_, 
                            agg_name=stat.colname_,
                            size=(600, 500),
                            title = "P(x|y) - Category Distribution By Target"
                        ),
                        title="Percentage"
                    )
                ]
            )
            
            tabs_y_x = Tabs(
                tabs = [
                    Panel(
                        child = chart_segment_group_count(
                            stat.p_y_x_,
                            group_name=stat.colname_, 
                            agg_name=stat.yname_,
                            size=(800, 500),
                            title = "P(y|x) - Event Count by Feature Categories"
                        ),
                        title="Count"
                    ),
                    Panel(
                        child = chart_segment_group_perc(
                            stat.p_y_x_,
                            group_name=stat.colname_, 
                            agg_name=stat.yname_,
                            size=(800, 500),
                            title = "P(x|y) - Event Rate by Feature Categories"
                        ),
                        title="Percentage"
                    )
                ]
            )
            
        elif dtype in ('Numeric'):
            tabs_x_y = Tabs(
                tabs = [
                    Panel(
                        child = chart_boxplot(
                            stat.p_x_y_['origin'],
                            xname=stat.colname_, 
                            yname=stat.yname_,
                            size=(400, 500),
                            title = "P(x|y) - Feature Distribution By Target"
                        ),
                        title="Origin"
                    ),
                    Panel(
                        child = chart_boxplot(
                            stat.p_x_y_['log'],
                            xname=stat.colname_, 
                            yname=stat.yname_,
                            size=(400, 500),
                            title = "P(x|y) - Feature Distribution By Target"
                        ),
                        title="Log"
                    )
                ]
            )
            
            tabs_y_x = chart_optbin_multiclass(
                stat.p_y_x_,
                ylabels = stat.ylabels_,
                xname=stat.colname_, 
                yname=stat.yname_,
                size=(1000, 500),
                title = "P(y|x) - Event Rate by Bucketized Feature"
            )
        
        return {
            'tabs_x_y' : tabs_x_y,
            'tabs_y_x' : tabs_y_x
        }

        
if 'dataframe' not in st.session_state:
    st.warning("No DataFrame uploaded, go back to profiling page first!")
else:    
    
    with st.sidebar:
        # about the feature and target selection
        selections = DataManager.getSelections()
        all_cols = DataManager.getColNames()
            
        targetname = st.selectbox(
            label = 'Target Or Catgorical Variable',
            options = all_cols,
            index = all_cols.index(selections['target'])
        )
        feature_cols = [col for col in DataManager.getTypedCols() if col != targetname]

        st.button(
            label = 'Calculate Univariate Correlation',
            type = 'primary',
            on_click = DataManager.calcUniCorr,
        )


    st.subheader("Feature - Target Correlation Chart")
    colname = st.selectbox(
        label = 'Feature',
        options = feature_cols,
        index = feature_cols.index(selections['feature'])
    )
    DataManager.rememberSelection(targetname, colname)
    try:
        figs = DataManager.getUniCorrPlot(colname)
    except KeyError as e:
        st.error(e)
    else:
        tabs_corr = st.tabs(['P(y|x)', 'P(x|y)'])
        with tabs_corr[0]:
            fig = figs['tabs_y_x']
            st.bokeh_chart(fig, use_container_width=True)
        with tabs_corr[1]:
            fig = figs['tabs_x_y']
            st.bokeh_chart(fig, use_container_width=True)


    st.subheader("Statistical Analysis")
    tabs_analysis = st.tabs(['Mutual Info', 'GLM', 'ANOVA'])
    with tabs_analysis[0]:
        if 'mutual_info' in st.session_state:
            chart_mi = chart_barchart(
                bar_arr = st.session_state['mutual_info'],
                max_bar = 100,
                title = 'Mutual Info Scores'
            )
            st.bokeh_chart(chart_mi, use_container_width=True)
        else:
            msg = st.session_state.get('mutual_info_err', "Click Calculate Button on the left")
            st.warning(msg)
            
    with tabs_analysis[1]:
        if 'glm_str' in st.session_state:
            glm_str = st.session_state['glm_str']
            st.code(glm_str, language = 'textile')
        else:
            msg = st.session_state.get('glm_err', "Click Calculate Button on the left")
            st.warning(msg)
            
    with tabs_analysis[2]:
        if 'anova_p' in st.session_state:
            chart_anova = chart_barchart(
                bar_arr = st.session_state['anova_p'],
                max_bar = 100,
                title = 'Anova F-stat PValues (-log10 scale) for Numerical Features'
            )
            st.bokeh_chart(chart_anova, use_container_width=True)
        else:
            msg = st.session_state.get('anova_err', "Click Calculate Button on the left")
            st.warning(msg)
            