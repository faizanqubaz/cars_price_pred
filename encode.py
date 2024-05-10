from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd


def encode_data(df):
    ct=ColumnTransformer(transformers=[
        ('ohe',OneHotEncoder(drop='first',sparse_output=False),['CarName','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem'])
    ],remainder='drop')

    ct_trans = ct.fit_transform(df)
    ct_columns = ct.get_feature_names_out()
    new_df = pd.DataFrame(ct_trans,columns=ct_columns)
    return new_df
