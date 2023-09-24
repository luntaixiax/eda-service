import json
import streamlit as st
st.set_page_config(layout='wide')
import pandas as pd
from typing import List, Tuple, Dict, Union, Literal
from bokeh.models import Panel, Tabs, DataTable, ColumnDataSource, TableColumn, StringFormatter, ScientificFormatter
from ModelingTools.Explore.profiling import _BaseStat, NumericStat, BinaryStat, OrdinalCategStat, NominalCategStat, TabularStat
from ModelingTools.Explore.plots import numeric_donut, chart_histogram, chart_categ_distr, categ_donut, plot_table_profiling
st.title('Exploratory Data Analysis')

def read_in_data(buf) -> pd.DataFrame:
    filename: str = buf.name
    if filename.endswith(".csv"):
        df = pd.read_csv(buf)
    elif filename.endswith(".parquet"):
        df = pd.read_parquet(buf)
    return df

def table_stat(stats: dict) -> DataTable:
    df = (
        pd.Series(stats)
        .reset_index(name='VALUE')
        .rename(columns = {'index' : 'METRIC'})
    )
    source = ColumnDataSource(df)
    data_table = DataTable(
        source=source, 
        columns=[
            TableColumn(
                field = 'METRIC',
                title = 'METRIC',
                sortable = False,
                formatter = StringFormatter(
                    font_style = 'bold'
                )
            ),
            TableColumn(
                field = 'VALUE',
                title = 'VALUE',
                sortable = False,
                formatter = ScientificFormatter(
                    precision = 5,
                    text_color = 'darkslategrey',
                )
            )
        ], 
        editable=False,
        index_position = None,
        # height = None,
        sizing_mode = 'stretch_both'
    )
    return data_table

class DataManager:
    @classmethod
    def init(cls):
        if 'standalone_config' not in st.session_state:
            st.session_state['standalone_config'] = {}
            st.session_state['tabular_stat'] = {
                'obj' : None,
                'json' : {},
                'html' : None
            }
    
    @classmethod
    def addData(cls, df: pd.DataFrame):
        # clear and replace the data
        st.session_state['dataframe'] = df
        
    @classmethod
    def deleteData(cls):
        if 'dataframe' in st.session_state:
            del st.session_state['dataframe']
        
    @classmethod
    def getData(cls) -> pd.DataFrame:
        return st.session_state['dataframe']
    
    @classmethod
    def getColNames(cls) -> List[str]:
        df = cls.getData()
        return df.columns.tolist()
    
    @classmethod
    def guessColType(cls, col: str) -> Literal['Binary', 'Ordinal', 'Nominal', 'Numeric']:
        vector = cls.getData()[col]
        if len(vector.dropna().unique()) == 2:
            return 'Binary'
        if pd.api.types.is_string_dtype(vector.dtype):
            return 'Nominal'
        elif pd.api.types.is_numeric_dtype(vector.dtype):
            if len(vector.unique()) < 5:
                return 'Ordinal'
            else:
                return 'Numeric'
        else:
            return 'Nominal'
        
    @classmethod
    def getValueList(cls, col: str) -> List:
        vector = cls.getData()[col]
        return vector.value_counts().index.tolist()
    
    
    @classmethod
    def getProfileSummaries(cls) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
        binaries, ordinals, nominals, nums = {}, {}, {}, {}
        for col, v in st.session_state['standalone_config'].items():
            if v['dtype'] == 'Binary':
                attr = v['stat'].to_dict()['attr']
                for k, a in attr.items():
                    if k in binaries:
                        binaries[k][col] = a
                    else:
                        binaries[k] = {col : a}
            if v['dtype'] == 'Ordinal':
                attr = v['stat'].to_dict()['attr']
                for k, a in attr.items():
                    if k in ordinals:
                        ordinals[k][col] = a
                    else:
                        ordinals[k] = {col : a}
            if v['dtype'] == 'Nominal':
                attr = v['stat'].to_dict()['attr']
                for k, a in attr.items():
                    if k in nominals:
                        nominals[k][col] = a
                    else:
                        nominals[k] = {col : a}
            if v['dtype'] == 'Numeric':
                attr = v['stat'].to_dict()['attr']
                for k, a in attr.items():
                    if k in nums:
                        nums[k][col] = a
                    else:
                        nums[k] = {col : a}
        return binaries, ordinals, nominals, nums
    
    @classmethod
    def updateProfile(cls, col:str, dtype:str, stat: _BaseStat):
        st.session_state['standalone_config'][col] = {
            'dtype' : dtype,
            'stat': stat
        }
        
    @classmethod
    def removeProfile(cls, col:str):
        if col in st.session_state['standalone_config']:
            del st.session_state['standalone_config'][col]
        
    @classmethod
    def getSavedProfile(cls, col:str):
        return st.session_state['standalone_config'].get(col, None)
    
    @classmethod
    def loadProfiles(cls, ts_dict: dict):
        # load tabular stat from dict
        ts = TabularStat.from_dict(ts_dict)
        for col, stat in ts.configs.items():
            dtype = {
                "NumericStat": "Numeric",
                "NominalCategStat": "Nominal",
                "OrdinalCategStat": "Ordinal",
                "BinaryStat": "Binary"
            }.get(stat.classname)
            cls.updateProfile(col, dtype = dtype, stat = stat)
        print(st.session_state['standalone_config'])
    
    @classmethod
    def getProfileFigs(cls, dtype:str, stat: _BaseStat) -> dict:
        # if col not in st.session_state['standalone_config']:
        #     raise KeyError("profile not exist, please click generate first")
        # config = st.session_state['standalone_config'][col]
        nst = stat
        if dtype == 'Nominal' or dtype == 'Ordinal':
            donut = categ_donut(
                total=nst.total_,
                missing=nst.missing_,
                size = (500, 400)
            )
            distr = chart_categ_distr(
                vcounts = nst.vcounts_,
                size = (500, 400)
            )
            return {
                'donut' : donut,
                'distr' : distr
            }
        elif dtype == 'Binary':
            donut = categ_donut(
                total=nst.total_,
                missing=nst.missing_,
                size = (500, 400)
            )
            distr = chart_categ_distr(
                vcounts = nst.binary_vcounts_,
                size = (500, 400)
            )
            return {
                'donut' : donut,
                'distr' : distr
            }
            
        elif dtype == 'Numeric':
            figs = {}
            
            donut = numeric_donut(
                nst.total_, 
                nst.missing_, 
                nst.zeros_, 
                nst.infs_pos_, 
                nst.infs_neg_, 
                nst.xtreme_,
                size = (600, 500)
            )
            figs['donut'] = donut
            
            tabs_desc_stat = [Panel(
                child = table_stat(nst.stat_descriptive_._asdict()),
                title="Origin"
            )]
            tabs_quant_stat = [Panel(
                child = table_stat(nst.stat_quantile_._asdict()),
                title="Origin"
            )]
            if nst.log_scale_:
                tabs_desc_stat.append(Panel(
                    child = table_stat(nst.stat_descriptive_log_._asdict()),
                    title="Log"
                ))
                tabs_quant_stat.append(Panel(
                    child = table_stat(nst.stat_quantile_log_._asdict()),
                    title="Log"
                ))

            tabs_hist = [Panel(
                child = chart_histogram(
                    bins_edges = nst.bin_edges_,
                    hists = nst.hist_,
                    quantile_stat=nst.stat_quantile_,
                    desc_stat=nst.stat_descriptive_,
                    title = 'Histogram for Valid Values',
                    size = (800, 500)
                ),
                title="Histogram - Origin"
            )]
            if nst.log_scale_:
                tabs_hist.append(Panel(
                    child = chart_histogram(
                        bins_edges = nst.bin_edges_log_,
                        hists = nst.hist_log_,
                        quantile_stat=nst.stat_quantile_log_,
                        desc_stat=nst.stat_descriptive_log_,
                        title = 'Histogram for Valid Values',
                        size = (800, 500)
                    ),
                    title="Statistics - Log"
                ))
                
            figs['quant'] = Tabs(tabs=tabs_quant_stat)
            figs['desc'] = Tabs(tabs=tabs_desc_stat)
            figs['hist'] = Tabs(tabs=tabs_hist)
                
            if nst.xtreme_method_ is not None:
                tabs_xtreme = [Panel(
                    child = table_stat(nst.xtreme_stat_._asdict()),
                    title="Origin"
                )]
                if nst.log_scale_:
                    tabs_xtreme.append(Panel(
                        child = table_stat(nst.xtreme_stat_log_._asdict()),
                        title="Log"
                    ))
                
                figs['xtreme'] = Tabs(tabs=tabs_xtreme)
                
            return figs
        
    @classmethod
    def registerTabular(cls, ts: TabularStat):
        st.session_state['tabular_stat'] = {
            'obj' : ts,
            'json' : ts.to_dict()
        }
        # save to html
        html_path = 'temp/feature_standalone_profile.html'
        try:
            plot_table_profiling(
                ts = ts,
                html_path = html_path
            )
        except Exception as e:
            st.json(ts.configs)
            st.error(e)
            html_path = None
        st.session_state['tabular_stat']['html'] = html_path
        
    @classmethod
    def getTabular(cls) -> TabularStat:
        return st.session_state['tabular_stat']['obj']
    
    @classmethod
    def getTabularJs(cls) -> dict:
        return st.session_state['tabular_stat']['json']
    
    @classmethod
    def getTabularHtml(cls) -> str:
        html_path = st.session_state['tabular_stat']['html']
        if html_path is not None:
            with open(html_path) as obj:
                buff = obj.read()
        else:
            buff = "<h>Nothing is here</h>"
        return buff
        


@st.cache_data
def get_or_train_binary_fsummary(col: str, pos_values: list, na_to_pos: bool, int_dtype: bool) -> _BaseStat:
    stat = BinaryStat(
        int_dtype = int_dtype,
        pos_values = pos_values,
        na_to_pos = na_to_pos
    )
    vector = DataManager.getData()[col]
    stat.fit(vector)
    return stat

@st.cache_data
def get_or_train_ordinal_fsummary(col: str, categories: List[Union[str, int]], int_dtype: bool) -> _BaseStat:
    stat =  OrdinalCategStat(
        int_dtype = int_dtype,
        categories = categories,
    )
    vector = DataManager.getData()[col]
    stat.fit(vector)
    return stat

@st.cache_data
def get_or_train_nominal_fsummary(col: str, max_categories: int, int_dtype: bool) -> _BaseStat:
    stat =  NominalCategStat(
        int_dtype = int_dtype,
        max_categories = max_categories,
    )
    vector = DataManager.getData()[col]
    stat.fit(vector)
    return stat
    
@st.cache_data
def get_or_train_numeric_fsummary(col: str, setaside_zero: bool, log_scale: bool, 
            xtreme_method: Literal['iqr', 'quantile'], bins: int) -> _BaseStat:
    stat = NumericStat(
        setaside_zero = setaside_zero,
        log_scale = log_scale,
        xtreme_method = xtreme_method,
        bins = bins
    )
    vector = DataManager.getData()[col]
    stat.fit(vector)
    return stat

def save_update_profile(col:str, dtype: str, stat: _BaseStat):
    # always update the profile
    DataManager.updateProfile(
        col = col,
        dtype = dtype,
        stat = stat
    )
    
def remove_profile(col:str):
    DataManager.removeProfile(col)

def train_tabular_stat():
    configs = {}
    for col, v in st.session_state['standalone_config'].items():
        configs[col] = v['stat']
        
    df = DataManager.getData()
    ts = TabularStat(
        configs = configs
    )
    ts.fit(df)
    DataManager.registerTabular(ts)
    
def clear_data():
    st.session_state['standalone_config'] = {}
    st.session_state['tabular_stat'] = {
        'obj' : None,
        'json' : {},
        'html' : None
    }
    DataManager.deleteData()

DataManager.init()

with st.sidebar:
    upload_file_buff = st.file_uploader(
        "Upload Dataset", 
        type = ['csv', 'parquet'], 
        key = 'upload_file',
        on_change = clear_data
    ) # clear when upload new
    if upload_file_buff is not None:
        df = read_in_data(upload_file_buff)
        DataManager.addData(df)


if 'dataframe' in st.session_state:
    #st.table(DataManager.getData().head())
    with st.sidebar:
        colname = st.selectbox(
            label = 'Feature',
            options = DataManager.getColNames(),
        )
    
    feature_stat_option_cols = st.columns(2)
    
    type_options = ['Binary', 'Nominal', 'Ordinal', 'Numeric']
    saved_config = DataManager.getSavedProfile(col = colname)
    if saved_config is not None:
        guessed_dtype = saved_config['dtype']
    else:
        st.info("Feature Never Seen, will guess Dtype on best effort")
        guessed_dtype = DataManager.guessColType(colname)
    
    with feature_stat_option_cols[0]:
        dtype_choice = st.selectbox(
            label = 'Feature Dtype',
            options = type_options,
            index = type_options.index(guessed_dtype),
        )
    
    with feature_stat_option_cols[1]:
        if dtype_choice == 'Binary':
            
            # important! must be same type to load back params, or will load to wrong class
            if guessed_dtype == dtype_choice and saved_config is not None:
                # saved value from last calculation
                binary_stat = saved_config['stat']
                int_dtype_v = binary_stat.int_dtype_
                na_to_pos_v = binary_stat.na_to_pos_
                pos_values_v = binary_stat.pos_values_
            else:
                # initial value
                int_dtype_v = False
                na_to_pos_v = False
                pos_values_v = None
            
            pos_values_select = st.multiselect(
                label = "Select values for class 1",
                options = DataManager.getValueList(col = colname),
                default = pos_values_v
            )
            cols_binary_ = st.columns([2,3])
            with cols_binary_[0]:
                int_dtype_check = st.checkbox(
                    label = 'Int Dtype',
                    value = int_dtype_v,
                )
            with cols_binary_[1]:
                na_to_pos_check = st.checkbox(
                    label = 'Missing value to class 1',
                    value = na_to_pos_v,
                )
            
            stat = get_or_train_binary_fsummary(
                col = colname,
                pos_values = pos_values_select,
                na_to_pos = na_to_pos_check,
                int_dtype = int_dtype_check
            )
            
        elif dtype_choice == 'Nominal':
            
            if guessed_dtype == dtype_choice and saved_config is not None:
                # saved value from last calculation
                nominal_stat = saved_config['stat']
                int_dtype_v = nominal_stat.int_dtype_
                max_categories_v = nominal_stat.max_categories_
            else:
                int_dtype_v = False
                max_categories_v = 25
                
            max_categories_inp = st.number_input(
                label = 'max # of categories',
                min_value = 3,
                max_value = 500,
                value = max_categories_v,
                step = 1
            )
            int_dtype_check = st.checkbox(
                label = 'Int Dtype',
                value = int_dtype_v,
            )
        
            stat = get_or_train_nominal_fsummary(
                col = colname,
                max_categories = max_categories_inp,
                int_dtype = int_dtype_check
            )
        
        elif dtype_choice == 'Ordinal':
            
            if guessed_dtype == dtype_choice and saved_config is not None:
                # saved value from last calculation
                ordinal_stat = saved_config['stat']
                int_dtype_v = ordinal_stat.int_dtype_
                categories_v = ordinal_stat.categories_
            else:
                int_dtype_v = False
                categories_v = None
            
            category_values_select = st.multiselect(
                label = "Rank Values for Ordinal Var",
                options = DataManager.getValueList(col = colname),
                default = categories_v
            )
            int_dtype_check = st.checkbox(
                label = 'Int Dtype',
                value = int_dtype_v,
            )
            stat = get_or_train_ordinal_fsummary(
                col = colname,
                categories = category_values_select,
                int_dtype = int_dtype_check
            )
            
        elif dtype_choice == 'Numeric':
            
            if guessed_dtype == dtype_choice and saved_config is not None:
                # saved value from last calculation
                numeric_stat = saved_config['stat']
                setaside_zero_v = numeric_stat.setaside_zero_
                log_scale_v = numeric_stat.log_scale_
                xtreme_method_v = numeric_stat.xtreme_method_
                bins_v = numeric_stat.bins_
            else:
                setaside_zero_v = False
                log_scale_v = False
                xtreme_method_v = 'iqr'
                bins_v = 100
                
            cols_num_ = st.columns(2)
            with cols_num_[0]:
                num_bins_inp = st.number_input(
                    label = '# of Bins for histogram',
                    min_value = 2,
                    max_value = 500,
                    value = bins_v,
                    step = 1
                )
                setaside_zero_check = st.checkbox(
                    label = 'Set aside Zeros',
                    value = setaside_zero_v,
                )
            with cols_num_[1]:
                xtreme_options = [None, 'iqr', 'quantile']
                xtreme_method_select = st.selectbox(
                    label = 'Xtreme Value Detection',
                    options = xtreme_options,
                    index = xtreme_options.index(xtreme_method_v)
                )
                logscale_check = st.checkbox(
                    label = 'Add Log Transform',
                    value = log_scale_v,
                )
            stat = get_or_train_numeric_fsummary(
                col = colname,
                setaside_zero = setaside_zero_check,
                log_scale = logscale_check,
                xtreme_method = xtreme_method_select,
                bins = num_bins_inp
            )
        
    
    with feature_stat_option_cols[0]:
        btn_cols = st.columns(2)
        with btn_cols[0]:
            st.button(
                label = 'Save/Update Feature Summary',
                type = 'primary',
                on_click = save_update_profile,
                kwargs = dict(
                    col = colname,
                    dtype = dtype_choice,
                    stat = stat
                )
            )
        with btn_cols[1]:
            st.button(
                label = 'Remove Feature Summary',
                type = 'secondary',
                on_click = remove_profile,
                kwargs = dict(
                    col = colname,
                )
            )
        
    try:
        profile_figs = DataManager.getProfileFigs(
            dtype = dtype_choice,
            stat = stat
        )
    except KeyError as e:
        st.info(e)
    else:
        if dtype_choice in ['Binary', 'Nominal', 'Ordinal']:
            categ_tabs = st.tabs(['Composite Donut', 'Category Distribution'])
            with categ_tabs[0]:
                st.bokeh_chart(figure = profile_figs['donut'], use_container_width =False)
            with categ_tabs[1]:
                st.bokeh_chart(figure = profile_figs['distr'], use_container_width =True)
                
        elif dtype_choice == 'Numeric':
            num_tabs = st.tabs(['Composit Donut', 'Histogram', 'Statistics', 'Xtremes'])
            with num_tabs[0]:
                st.bokeh_chart(figure = profile_figs['donut'], use_container_width =False)
            with num_tabs[1]:
                st.bokeh_chart(figure = profile_figs['hist'], use_container_width =True)
            with num_tabs[2]:
                col_nums_stat_tb = st.columns(2)
                with col_nums_stat_tb[0]:
                    st.bokeh_chart(figure = profile_figs['quant'], use_container_width =False)
                with col_nums_stat_tb[1]:
                    st.bokeh_chart(figure = profile_figs['desc'], use_container_width =False)
            with num_tabs[3]:
                if profile_figs.get('xtreme') is not None:
                    st.bokeh_chart(figure = profile_figs['xtreme'], use_container_width =False)
        
    
    
    bottom_tabs = st.tabs(['Summary', 'I/O'])
    with bottom_tabs[0]:
        st.button(
            label = 'Train Standalone Feature Stat',
            type = 'primary',
            on_click = train_tabular_stat,
        )
        
        binaries_summary, ordinals_summary, nominals_summary, nums_summary = DataManager.getProfileSummaries()
        summ_cols = st.columns([2, 3])
        with summ_cols[0]:
            st.text("Binary Varaibles Summary")
            st.dataframe(binaries_summary, use_container_width = True)
            st.text("Nominal Categorical Varaibles Summary")
            st.dataframe(nominals_summary, use_container_width = True)
        with summ_cols[1]:
            st.text("Numerical Varaibles Summary")
            st.dataframe(nums_summary, use_container_width = True)
            st.text("Ordinal Categorical Summary")
            st.dataframe(ordinals_summary, use_container_width = True)
            
        #st.json(st.session_state['standalone_config'])
        
    with bottom_tabs[1]:
        io_cols = st.columns(2)
        with io_cols[0]:
            upload_stat_js_buff = st.file_uploader(
                label = 'Upload Tabular Stat from JSON',
                type = ['json'],
                accept_multiple_files = False,
            )
            st.info("Note that only Feature Dtype and Parameters will be loaded")
            if upload_stat_js_buff is not None:
                ts_dict = json.load(upload_stat_js_buff)
                DataManager.loadProfiles(ts_dict)
        
        with io_cols[1]:
            st.text("Download Tabular Stat as")
            download_cols = st.columns([2, 5])
            with download_cols[0]:
                st.download_button(
                    label = "JSON",
                    file_name = "feature_standalone_profile.json",
                    mime = "application/json",
                    data = json.dumps(DataManager.getTabularJs(), indent = 4),
                )
            with download_cols[1]:
                st.download_button(
                    label = "HTML",
                    file_name = "feature_standalone_profile.html",
                    mime = "application/html",
                    data = DataManager.getTabularHtml(),
                )