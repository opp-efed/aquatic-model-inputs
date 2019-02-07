import numpy as np
import pandas as pd
from scipy import stats

from utilities import fields, calculate_uslels
from parameters import max_horizons, hydro_soil_groups, uslep_values, bins, depth_bins


def process_soils(soils):
    """  Identify component to be used for each map unit """

    # Adjust soil data values
    soils.loc[:, 'orgC'] /= 1.724  # oc -> om
    soils.loc[:, ['fc', 'wp']] /= 100.

    # Isolate unique map unit/component pairs and select major component with largest area (comppct)
    components = soils[['mukey', 'cokey', 'major_component', 'component_pct']].drop_duplicates(['mukey', 'cokey'])
    components = components[components.major_component == 'Yes']
    components = components.sort_values('component_pct', ascending=False)
    components = components[~components.mukey.duplicated()]
    soils = components[['mukey', 'cokey']].merge(soils, on=['mukey', 'cokey'], how='left')

    # Sort table by horizon depth and get horizon information
    soils = soils.sort_values(['cokey', 'horizon_top'])
    soils['thickness'] = soils['horizon_bottom'] - soils['horizon_top']
    soils['horizon_num'] = soils.groupby('cokey').cumcount() + 1
    soils['n_horizons'] = soils.groupby('cokey').size()

    # Extend columns of data for multiple horizons
    horizon_data = soils.set_index(['cokey', 'horizon_num'])[fields.fetch('horizontal') + ['kwfact']]
    horizon_data = horizon_data.unstack().sort_index(1, level=1)
    horizon_data.columns = ['_'.join(map(str, i)) for i in horizon_data.columns]
    soils = soils.drop_duplicates().merge(horizon_data, left_on='cokey', right_index=True)

    # New HSG code - take 'max' of two versions of hsg
    hsg_to_num = {hsg: i + 1 for i, hsg in enumerate(hydro_soil_groups)}
    num_to_hsg = {v: k.replace("/", "") for k, v in hsg_to_num.items()}
    soils['hydro_group'] = soils[['hydro_group', 'hydro_group_dominant']].applymap(
        lambda x: hsg_to_num.get(x)).max(axis=1).fillna(-1).astype(np.int32)
    soils['hsg_letter'] = soils['hydro_group'].map(num_to_hsg)

    # Calculate USLE LS and P values
    soils['uslels'] = soils.apply(calculate_uslels, axis=1)
    soils['uslep'] = np.array(uslep_values)[np.int16(pd.cut(soils.slope, bins['slope'], labels=False))]

    # Select kwfact
    soils['kwfact'] = soils.kwfact_1
    soils.loc[soils.kwfact == 9999, 'kwfact'] = soils.kwfact_2
    soils.loc[soils.kwfact == 9999, 'kwfact'] = np.nan
    soils.loc[(soils.horizon_top_1 > 1) | (soils.horizon_top_1 < 0), 'kwfact'] = np.nan
    soils.loc[soils.desgnmaster_1 == 'R', 'kwfact'] = np.nan

    return soils


def aggregate_soils(soils):
    from parameters import bins

    # Sort data into bins
    out_data = [soils.hsg_letter]
    for field, field_bins in bins.items():
        labels = [field[:2 if field == "slope" else 1] + str(i) for i in range(1, len(field_bins))]
        sliced = pd.cut(soils[field].fillna(0), field_bins,
                        labels=labels, right=False, include_lowest=True)
        out_data.append(sliced.astype("str"))
    soil_agg = pd.concat(out_data, axis=1)

    # Create aggregation key
    soils['aggregation_key'] = \
        soil_agg['hsg_letter'] + soil_agg['slope'] + soil_agg['orgC_5'] + soil_agg['sand_5'] + soil_agg['clay_5']

    # Group by aggregation key and take the mean of all properties except HSG, which will use mode
    grouped = soils.groupby('aggregation_key')
    averaged = grouped.mean().reset_index()
    hydro_group = grouped['hydro_group'].agg(lambda x: stats.mode(x)[0][0]).to_frame().reset_index()
    del averaged['hydro_group']
    soils = averaged.merge(hydro_group, on='aggregation_key')

    return soils


def depth_weight_soils(soils):
    # Get the root name of depth weighted fields
    depth_fields = {f.split("_")[0] for f in fields.fetch('depth_weight')}
    depth_weighted = []

    # Generate weighted columns for each bin
    for bin_top, bin_bottom in zip([0] + list(depth_bins[:-1]), list(depth_bins)):
        bin_table = np.zeros((soils.shape[0], len(depth_fields)))

        # Perform depth weighting on each horizon
        for i in range(max_horizons):
            # Set field names for horizon
            top_field, bottom_field = 'horizon_top_{}'.format(i + 1), 'horizon_bottom_{}'.format(i + 1)
            value_fields = ["{}_{}".format(f, i + 1) for f in depth_fields]

            # Adjust values by bin
            horizon_bottom, horizon_top = soils[bottom_field], soils[top_field]
            overlap = (horizon_bottom.clip(upper=bin_bottom) - horizon_top.clip(lower=bin_top)).clip(0)
            ratio = (overlap / (horizon_bottom - horizon_top)).fillna(0)
            bin_table += soils[value_fields].fillna(0).mul(ratio, axis=0).values

        # Add columns
        bin_table = \
            pd.DataFrame(bin_table, columns=["{}_{}".format(f, bin_bottom) for f in depth_fields])
        depth_weighted.append(bin_table)

    # Clear old fields and append new one
    for field in fields.fetch('horizons_expanded'):
        print(field)
        del soils[field]
    soils = pd.concat([soils] + depth_weighted, axis=1)
    return soils
