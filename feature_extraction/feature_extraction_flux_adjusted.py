import numpy as np
import pandas as pd
from feature_extraction import aggregation

ID = 'object_id'
target = 'target'

def feature_extraction(df_meta, df):
    # Adjust flux to magnitude, which equals to flux * photoz * photoz
    df = df.merge(df_meta[[ID, 'distmod', 'hostgal_photoz']], on=ID, how='left')
    df = df[df['distmod'].notnull()]
    df['flux'] *= df['hostgal_photoz'] * df['hostgal_photoz']
    df['flux_err'] *= df['hostgal_photoz'] * df['hostgal_photoz']
	
    passband_agg = aggregation(df, [ID, 'passband'])
    object_agg = aggregation(df, ID)
    return passband_agg, object_agg

if 'name' == 'main':
    train_meta = pd.read_csv('../input/training_set_metadata.csv')
    train = pd.read_csv('../input/training_set.csv')
    train_passband_agg, train_object_agg = feature_extraction(train_meta, train)
    train_passband_agg.to_csv('train_passband_aggregation_flux_adjusted.csv', index=False)
    train_object_agg.to_csv('train_object_aggregation_flux_adjusted.csv', index=False)

    test_meta = pd.read_csv('../input/test_set_metadata.csv')
    test = pd.read_csv('../input/test_set.csv')
    test_passband_agg, test_object_agg = feature_extraction(test_meta, test)
    test_passband_agg.to_csv('test_passband_aggregation_flux_adjusted.csv', index=False)
    test_object_agg.to_csv('test_object_aggregation_flux_adjusted.csv', index=False)
