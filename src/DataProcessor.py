import numpy as np
import pandas as pd
import re


class DataProcessor():
    """ 
    Data processing class for loading, cleaning and extracting features
    Class is initialied with the path to the data:str , and a target: str

    """

    def __init__(self, path: str, target: str):
        self.path = path
        self.target = target

    def load_data(self):
        """ 
        ## Summary
        imports data from self.path csv file and stores as self.data, low memory is set to false due to coloumns containing
        multiple data types

        ### Returns: 
        Stores self and stores dataframe as self.data attribute


        """

        self.data = pd.read_csv(self.path, low_memory=False)
        return self

    def clean_data(self):
        """
        ### Summary: 

        Cleans and prepares data. Intatiates as self.clean_data 

        #### Returns:
           self
        """

        df = self.data

        def find_cols_with_missing_data(_df, threshold):
            """Returns a list of columns with missing data over specified threshold"""
            return [col for col in _df.columns if _df[col].isnull().sum() > (_df.shape[0]*threshold)]

        def find_cols_with_single_val(_df):
            """ returns a list of columns with single unique values"""
            non_uniuqe_vals = _df.nunique(dropna=True) == 1
            return [non_uniuqe_vals.index[i] for i, col in enumerate(non_uniuqe_vals) if col == True]

        missing_data_cols = find_cols_with_missing_data(df, threshold=0.5)
        non_uniq_cols = find_cols_with_single_val(df)

        data_leak_cols = ['out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
                          'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
                          'funded_amnt', 'funded_amnt_inv', 'issue_d', 'pymnt_plan', 'last_pymnt_d', 'last_pymnt_amnt', 'int_rate']

        non_useful_cols = ['title', 'emp_title',
                           'last_credit_pull_d', 'earliest_cr_line', 'addr_state']
        edited_cols = ['verification_status', 'sub_grade', 'id',
                       'member_id', 'verification_status', 'zip_code', 'addr_state']

        columns_to_drop = [*missing_data_cols, *data_leak_cols,
                           *edited_cols, *non_uniq_cols, *non_useful_cols]

        ordinal_grade_type = pd.CategoricalDtype(
            categories=sorted(df.grade.dropna().unique()), ordered=True)

        self.clean_data = (
            df
            .iloc[:-2]
            .assign(purpose=lambda df_: df_.purpose.astype('category'),
                    home_ownership=lambda df_: df_.home_ownership.astype(
                        'category'),
                    grade=lambda df_: df_.grade.astype(ordinal_grade_type),
                    emp_length=lambda df_: df_.emp_length.str.replace(
                        '\D', '', regex=True).dropna().astype(int),
                    verification_status=lambda df_: df_.verification_status.astype(
                        'category'),
                    revol_util=lambda df_: df_.revol_util.str.replace(
                        '%', '').astype(float)
                    )
            .drop(columns=columns_to_drop)
            .dropna()
            .query("loan_status == 'Fully Paid' or loan_status == 'Charged Off'")
            .replace(
                {
                    "loan_status": {"Fully Paid": 1, "Charged Off": 0},
                    "term": {" 36 months": 0, " 60 months": 1}
                }
            )
        )
        return self

    def processes_dummies(self, cols=None):
        """ 
        ## Summary: 
        generates dummies from categorical data types in self.clean_data and concatenates with original dataframe
        (self.clean_data). Drops cols from dataframe. 

        The cols parameter is optional. Function checks that no input was given, if ==None, then extracts all categorical features. If cols list is given, then that list is used instead

        ### Returns:
        self, and instantiates self.processed_data


        """
        if cols == None:
            cols = self.clean_data.select_dtypes('category').columns
        else:
            cols = cols

        dummy_df = pd.get_dummies(self.clean_data[cols])
        self.processed_data = (
            pd.concat([self.clean_data, dummy_df], axis=1)
            .drop(columns=cols)
        )
        return self

    def extract_features_and_target(self):
        """
        ### Summary
        extracts feature names from self.processed_data
        Instatiates self.features


        #### Returns:
            self
        """

        self.feature_names = (self.processed_data
                              .drop(columns=self.target).columns)

        self.feature_data = self.processed_data[self.feature_names]
        self.target_data = self.processed_data[self.target]
        return self
