import os

import numpy as np
import pandas as pd
# import dask.dataframe as dd

from parameters import states_nhd, cell_size, vpus_nhd, chunk_size
from paths import condensed_soil_path, nhd_path, met_attributes_path, combo_path, crop_group_path, crop_dates_path, \
    crop_params_path, gen_params_path, irrigation_path
from utilities import fields, read_dbf


def combinations_new(region, year):
    """ Not yet implemented pending new combination files """
    for state in states_nhd[region]:
        combo_file = combo_path.format(region, state, year)
        combo_table = pd.DataFrame(dtype=np.int32, **np.load(combo_file))
        matrix = combo_table if matrix is None else pd.concat([matrix, combo_table], axis=0)

    # Column names get written as binary by spatial_overlay
    matrix.columns = [f.decode('ascii') for f in matrix.columns.values]

    # Combine duplicate rows and convert counts to areas
    matrix = matrix.groupby(['gridcode', 'weather_grid', 'cdl', 'mukey']).sum().reset_index()
    matrix['area'] *= cell_size ** 2

    return matrix


def combinations(region, year):
    combos = pd.DataFrame(**np.load(combo_path.format(region, year)), dtype=np.int32)
    return combos


def crop_data():
    # Merge all crop-related data tables
    data = pd.read_csv(crop_group_path) \
        .merge(pd.read_csv(crop_params_path).rename(columns={'cdl': 'alias'}), on='alias', how='left') \
        .merge(pd.read_csv(crop_dates_path), on=['cdl', 'alias'], how='left') \
        .merge(pd.read_csv(irrigation_path).rename(columns={'cdl': 'alias'}), on=['alias', 'state'], how='left') \
        .merge(pd.read_csv(gen_params_path), on='gen_class', how='left', suffixes=('_cdl', '_gen'))

    # Convert to dates
    for field in fields.fetch('CropDates'):
        data[field] = (pd.to_datetime(data[field], format="%d-%b") - pd.to_datetime("1900-01-01")).dt.days

    return data


def nhd_params(region):
    gridcodes_path = \
        os.path.join(nhd_path.format(vpus_nhd[region], region), "NHDPlusCatchment", "featureidgridcode.dbf")
    return read_dbf(gridcodes_path)[['featureid', 'gridcode']].rename(columns={"featureid": "comid"})


def met():
    met_data = pd.read_csv(met_attributes_path)
    del met_data['weather_grid']
    met_data = met_data.rename(columns={"stationID": 'weather_grid'})  # these combos have old weather grids?
    del met_data['state']  # for now, use state parameter from SSURGO
    return met_data


def soils(region):
    horizon_tables, component_tables = [], []
    for state in states_nhd[region]:
        try:
            horizon_table = os.path.join(condensed_soil_path, '{}_horizon.csv'.format(state.upper()))
            horizons = pd.read_csv(horizon_table, index_col=0)
            component_table = os.path.join(condensed_soil_path, '{}_params.csv'.format(state.upper()))
            components = pd.read_csv(component_table, index_col=0)
            components['state'] = state
            component_tables.append(components)
            horizon_tables.append(horizons)
        except FileNotFoundError:
            print("No soils data found for {}".format(state))
    chorizon_table = pd.concat(horizon_tables, axis=0)
    soil_params = pd.concat(component_tables, axis=0)

    # Change field names
    chorizon_table.rename(columns=fields.convert, inplace=True)
    soil_params.rename(columns=fields.convert, inplace=True)
    combined = pd.merge(chorizon_table, soil_params, on='cokey', how='outer')
    return combined


if __name__ == "__main__":
    __import__('1_scenarios_and_recipes').main()
