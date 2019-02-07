import os

import numpy as np
import pandas as pd

from parameters import states_nhd, cell_size, vpus_nhd
from paths import condensed_soil_path, nhd_path, met_attributes_path, combo_path
from utilities import fields, read_dbf


def combinations(region, year):
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


def crop_data():
    """ Merge all parameter tables linked to crop class """

    # Since crop data table is read in more than 1 script, reading this table is done in utilities.CropMatrix
    from utilities import fields, crops as crop_data

    def read_dates(series):
        return (pd.to_datetime(series, format="%d-%b") - pd.to_datetime("1900-01-01")).dt.days

    # Parse crop dates
    for field_stem in fields.fetch("CropDates"):
        if "plant" in field_stem:
            for var in "ab":
                # Identify harvest field corresponding to plant field
                plant_field = field_stem + "_" + var
                harvest_field = plant_field.replace("plant", "harvest")

                # Convert string dates to datetime
                crop_data[plant_field] = read_dates(crop_data[plant_field])
                crop_data[harvest_field] = read_dates(crop_data[harvest_field])
                # Add a year if harvest happens before plant (e.g. plant - Nov 1, harvest Feb 3)
                crop_data.loc[crop_data[harvest_field] < crop_data[plant_field], harvest_field] += 365.
    return crop_data


def gridcodes(region):
    gridcodes_path = \
        os.path.join(nhd_path.format(vpus_nhd[region], region), "NHDPlusCatchment", "featureidgridcode.dbf")
    return read_dbf(gridcodes_path)[['featureid', 'gridcode']].rename(columns={"featureid": "comid"})


def met():
    met_data = pd.read_csv(met_attributes_path)
    del met_data['state']  # for now, use state parameter from SSURGO
    return met_data


def soils(region):
    horizon_tables, component_tables = [], []
    for state in states_nhd[region]:
        horizon_table = os.path.join(condensed_soil_path, '{}_horizon.csv'.format(state.upper()))
        horizons = pd.read_csv(horizon_table, index_col=0)
        component_table = os.path.join(condensed_soil_path, '{}_params.csv'.format(state.upper()))
        components = pd.read_csv(component_table, index_col=0)
        components['state'] = state
        component_tables.append(components)
        horizon_tables.append(horizons)
    chorizon_table = pd.concat(horizon_tables, axis=0)
    soil_params = pd.concat(component_tables, axis=0)

    # Change field names
    chorizon_table.rename(columns=fields.convert, inplace=True)
    soil_params.rename(columns=fields.convert, inplace=True)
    combined = pd.merge(chorizon_table, soil_params, on='cokey', how='outer')
    return combined
