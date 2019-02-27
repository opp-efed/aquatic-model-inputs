import modify
import numpy as np
import pandas as pd
import read
import write

from utilities import fields
from parameters import nhd_regions


def update_fields(depth_weight, n_horizons=0):
    if depth_weight:
        fields.expand('depth')
    else:
        fields.expand('horizon', n_horizons)


def create_recipes(combinations, watershed_data):
    combinations = combinations.merge(watershed_data, on='gridcode')
    comids = combinations.pop('comid')
    recipes = combinations[['scenario_id', 'area']]

    # Get the first and last row corresponding to each comid
    first_rows = np.hstack(([0], np.where(comids[:-1] != comids[1:])[0] + 1))
    last_rows = np.hstack((first_rows[1:], [comids.size]))
    recipe_map = np.vstack((comids[first_rows], first_rows, last_rows)).T

    return recipes, recipe_map


def create_scenarios(combinations, met_data, crop_data, soil_data, depth_weight, n_horizons, mode):
    # Join remaining input tables
    scenarios = combinations.merge(soil_data, how="left", on="soil_id")
    scenarios = scenarios.merge(crop_data, how="left", on=['cdl', 'alias', 'state'])
    scenarios = scenarios.merge(met_data, how="left", on="weather_grid")

    # Perform scenario attribution
    scenarios = modify.process_scenarios(scenarios)

    # Select output fields
    out_fields = fields.fetch(mode + '_scenario')

    return scenarios[out_fields + ['alias']]


def update_combinations(all_combos, new_combos):
    # Watershed information is not retained for scenarios
    del new_combos['gridcode']

    # Append new scenarios to running list
    all_combos = pd.concat([all_combos, new_combos], axis=0) if all_combos is not None else new_combos

    # Combine duplicate scenarios by adding areas
    all_combos = all_combos.groupby([f for f in all_combos.columns if f != 'area']).sum().reset_index()

    return all_combos


def scenarios_and_recipes(regions, years, aggregate, exclude, depth_weight, make_recipes, mode):
    # Read data indexed to weather grid
    met_data = read.met()

    # Read data indexed to crop
    crop_data = read.crop_data()

    # Soils, watersheds and combinations are broken up by NHD region
    for region in regions:

        # Initialize scenarios
        all_combinations = None

        # Read data indexed to watershed
        watershed_data = read.nhd_params(region)

        # Read data indexed to soil and process
        soil_data = read.soils(region)
        soil_data, n_horizons = modify.process_soils(soil_data, aggregate, depth_weight)

        # Update fields object
        update_fields(depth_weight, n_horizons)

        for year in years:
            print("Processing Region {}, {}".format(region, year))

            # Read met/crop/land cover/soil/watershed combinations and process
            combinations = read.combinations(region, year)
            combinations = \
                modify.process_combinations(combinations, crop_data, soil_data, met_data, aggregate, exclude)

            # Append combinations for year to running table
            all_combinations = update_combinations(all_combinations, combinations)

            # Create recipes
            if make_recipes:
                recipes, recipe_map = create_recipes(combinations, watershed_data)
                write.recipes(recipes, recipe_map)

        # Join combinations with data tables and perform additional attribution
        scenarios = create_scenarios(all_combinations, met_data, crop_data, soil_data, depth_weight, n_horizons, mode)

        # Write scenarios to file, making random selections by crop for PWC
        if mode == 'pwc':
            for crop_name, crop_scenarios in modify.select_pwc_scenarios(scenarios, n_horizons, crop_data):
                write.scenarios(crop_scenarios, mode, region, crop_name)
        else:
            write.scenarios(scenarios, mode, region)


def main():
    modes = ('pwc',)  # pwc and/or sam
    regions = nhd_regions
    regions = ['07']
    years = range(2010, 2014)

    for mode in modes:
        # Automatically adjust run parameters for pwc or sam
        aggregate = (mode == 'sam')
        exclude = (mode == 'pwc')
        depth_weight = (mode == 'sam')
        make_recipes = (mode == 'sam')

        scenarios_and_recipes(regions, years, aggregate, exclude, depth_weight, make_recipes, mode)


if __name__ == "__main__":
    main()
