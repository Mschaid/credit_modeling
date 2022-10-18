import numpy as np
import pandas as pd
import re


class DataProcessor():

    def __init__(self, path, target):
        self.path = path
        self.target = target

    def load_data(self):
        self.data = pd.read_csv(self.path)
        return self

    def clean_data(self):
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

    def processes_dummies(self, cols):
    dummy_df = pd.get_dummies(self.clean_data[cols])
    self.processed_data = (
        pd.concat([self.clean_data, dummy_df], axis=1)
        .drop(columns=cols)
    )
    return self