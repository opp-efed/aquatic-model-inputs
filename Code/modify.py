import numpy as np
import pandas as pd
from scipy import stats

from utilities import fields
from parameters import max_horizons, hydro_soil_groups, uslep_values, bins, depth_bins, hsg_cultivated, \
    hsg_non_cultivated, null_curve_number, usle_m_vals

from parameters import pwc_selection_field as crop_field
from parameters import pwc_sample_size as sample_size


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

    return soils.rename(columns={'aggregation_key': 'soil_id'})


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
        del soils[field]
    soils = pd.concat([soils] + depth_weighted, axis=1)
    return soils


def process_combinations(combos, crop_data, soil_data, met_data, aggregate, exclude):
    # Split double-cropped classes into individual scenarios
    combos = combos.merge(crop_data[['cdl', 'alias']].drop_duplicates(), on='cdl', how='left')

    # Aggregate scenarios, if necessary
    if aggregate:
        scenarios = combos.merge(soil_data[['mukey', 'soil_id']], how="left", on="mukey")
        del scenarios['mukey']
        combos = scenarios.groupby(['gridcode', 'weather_grid', 'cdl', 'soil_id']).sum().reset_index()
    else:
        combos = combos.rename(columns={'mukey': 'soil_id'})

    # Scenarios are only selected for PWC where data is complete
    original_cols = combos.columns
    available_crops = crop_data.dropna(subset=fields.fetch('CropDates'), how='all')[['cdl', 'alias', 'state']]

    if exclude:
        combos = combos.merge(soil_data[['soil_id', 'state']], on='soil_id', how='inner')
        combos = combos.merge(available_crops, on=['cdl', 'alias', 'state'], how='inner')

    return combos[original_cols].astype(np.int32)


def process_soils(soils, depth_weight, aggregate):
    """  Identify component to be used for each map unit """

    # Adjust soil data values
    soils.loc[:, 'orgC'] /= 1.724  # oc -> om
    soils.loc[:, ['fc', 'wp']] /= 100.
    soils.loc[(soils.kwfact == 0) | (soils.kwfact == 9999), 'kwfact'] = np.nan

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
    soils = soils.sort_values('horizon_num', ascending=False)  # n_horizons
    n_horizons = soils['horizon_num'].max()

    # Extend columns of data for multiple horizons
    horizon_data = soils.set_index(['cokey', 'horizon_num'])[fields.fetch('horizontal') + ['kwfact']]
    horizon_data = horizon_data.unstack().sort_index(1, level=1)
    horizon_data.columns = ['_'.join(map(str, i)) for i in horizon_data.columns]
    soils = soils.drop_duplicates(['mukey', 'cokey']).merge(horizon_data, left_on='cokey', right_index=True)

    # New HSG code - take 'max' of two versions of hsg
    hsg_to_num = {hsg: i + 1 for i, hsg in enumerate(hydro_soil_groups)}
    num_to_hsg = {v: k.replace("/", "") for k, v in hsg_to_num.items()}
    soils['hydro_group'] = soils[['hydro_group', 'hydro_group_dominant']].applymap(
        lambda x: hsg_to_num.get(x)).max(axis=1).fillna(-1).astype(np.int32)
    soils['hsg_letter'] = soils['hydro_group'].map(num_to_hsg)

    # Calculate USLE LS and P values
    m = usle_m_vals[np.int16(pd.cut(soils.slope.values, bins['usle_m'], labels=False))]
    sine_theta = np.sin(np.arctan(soils.slope / 100))  # % -> sin(rad)
    soils['uslels'] = (soils.slope_length / 72.6) ** m * (65.41 * sine_theta ** 2. + 4.56 * sine_theta + 0.065)
    soils['uslep'] = np.array(uslep_values)[np.int16(pd.cut(soils.slope, bins['slope'], labels=False))]

    # Select kwfact
    soils['kwfact'] = soils[["kwfact_{}".format(i + 1) for i in range(n_horizons)]].bfill(1).iloc[:, 0]

    if depth_weight:
        soils = depth_weight_soils(soils)
    if aggregate:
        soils = aggregate_soils(soils)
    else:
        soils = soils.rename(columns={'mukey': 'soil_id'})

    return soils, n_horizons


def process_scenarios(scenarios):
    # Emergence 7 days after planting, maturity halfway between plant and harvest
    scenarios['emergence_begin'] = np.int32(scenarios.plant_begin + 7)
    scenarios['maxcover_begin'] = np.int32((scenarios.plant_begin + scenarios.harvest_begin) / 2)
    scenarios['emergence_end'] = np.int32(scenarios.plant_end + 7)
    scenarios['maxcover_end'] = np.int32((scenarios.plant_end + scenarios.harvest_end) / 2)

    # Process curve number
    scenarios['cn_ag'] = null_curve_number
    scenarios['cn_fallow'] = null_curve_number
    for cultivated, soil_groups in enumerate((hsg_non_cultivated, hsg_cultivated)):
        for group_num, group_name in enumerate(soil_groups):
            sel = (scenarios.hydro_group == group_num + 1) & (scenarios.cultivated_cdl == cultivated)
            for var in 'ag', 'fallow':
                scenarios.loc[sel, 'cn_' + var] = scenarios.loc[sel, 'cn_{}_{}'.format(var, group_name)]

    # Deal with maximum rooting depth
    scenarios['amxdr'] = scenarios[['amxdr', 'root_zone_max']].min(axis=1)

    # Recode for old met station ids
    # scenarios['weather_grid'] = scenarios['stationID']

    # Temporary defaults
    scenarios['crop_prac'] = "C_CR"
    scenarios['season'] = 1
    scenarios['sfac'] = 0.274

    # Create a CDL/weather/soil identifier
    scenarios['scenario_id'] = 's' + scenarios.soil_id.astype("str") + \
                               'w' + scenarios.weather_grid.astype("str") + \
                               'lc' + scenarios.cdl.astype("str")

    return scenarios


def select_pwc_scenarios(scenarios, n_horizons, crop_data):
    # Remove scenarios that have missing or invalid data
    qc_table = fields.perform_qc(scenarios)
    horizon_fields = [f for f in fields.fetch('horizontal') if f in fields.fetch('pwc_scenario')]
    other_fields = [f for f in fields.fetch('pwc_scenario') if f not in fields.fetch('horizontal')]
    check = qc_table[horizon_fields].replace(2, np.nan)
    valid_horizons = n_horizons - (check.isnull().sum(axis=1) / (len(horizon_fields) / n_horizons))
    invalid_horizons = (valid_horizons != scenarios.horizon_num)
    invalid_other = qc_table[other_fields].max(axis=1) == 2
    burn = invalid_horizons + invalid_other

    # scenarios[(valid_horizons != scenarios.horizon_num)].to_csv("burn1.csv")
    # qc_table[(valid_horizons != scenarios.horizon_num)].to_csv("burn2.csv")

    # Create a report of why scenarios were excluded
    report = (qc_table[other_fields] == 2).sum(axis=0)
    report['n_scenarios'] = scenarios.shape[0]
    report['removed_scenarios'] = burn.sum()
    report['horizons'] = invalid_horizons.sum()
    report = report.sort_values(ascending=False)
    a = scenarios[qc_table.amxdr == 2]['weather_grid'].drop_duplicates().values
    print(" Or stationID = ".join(map(str, a)))
    scenarios = scenarios[~burn]

    # Create a 'metadata' table of scenario counts
    counts = np.array(np.unique(scenarios[crop_field][~np.isnan(scenarios[crop_field])], return_counts=True)).T
    table = pd.DataFrame(counts, columns=[crop_field, 'n_scenarios'], dtype=np.int32)
    table['sample_size'] = table.n_scenarios.where(table.n_scenarios < sample_size, sample_size)
    yield 'meta', table

    # Randomly sample from each crop group and save the sample
    crop_groups = crop_data[[crop_field, crop_field + '_desc']].drop_duplicates().values
    for crop, crop_name in crop_groups:
        sample = scenarios.loc[scenarios[crop_field] == crop]
        if sample.shape[0] > sample_size:
            sample = sample.sample(sample_size)
        if not sample.empty:
            yield '{}_{}'.format(crop, crop_name), sample


if __name__ == "__main__":
    __import__('1_scenarios_and_recipes').main()
