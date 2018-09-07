import os
import numpy as np
import pandas as pd
import datetime as dt


def get_scenario_counts(matrix, sample_size):
    counts = np.array(np.unique(matrix.gen_class[~np.isnan(matrix.gen_class)], return_counts=True)).T
    table = pd.DataFrame(counts, columns=['gen_class', 'n_scenarios'], dtype=np.int32)
    table['sample_size'] = table.n_scenarios.where(table.n_scenarios < sample_size, sample_size)
    return table


def sample_scenarios(matrix, _class, sample_size):
    sample = matrix.loc[matrix['gen_class'] == _class]
    if sample.shape[0] > sample_size:
        sample = matrix.sample(sample_size)
    return sample


def fetch_metfiles(grid_ids, weather_cube, out_path, overwrite=False):
    for grid_id in grid_ids:
        outfile_path = os.path.join(out_path, grid_id)
        if not os.path.exists(outfile_path):
            weather_cube.fetch(grid_id).to_csv(outfile_path)


def main():
    from paths import scenario_matrix_path, metfile_path, pwc_scenario_path, pwc_metfile_path
    from utilities import fields, crops, WeatherCube

    # Set run parameters
    regions = ['07']
    years = [2010]
    sample_size = 100
    overwrite_met = False

    # Initialize output fields
    fields.expand('horizon')
    cultivated_crops = crops[crops.cultivated_gen == 1][['gen_class', 'gen_class_desc']].values

    weather = WeatherCube(metfile_path)

    # Iterate through each region and year
    for region in regions:
        for year in years:

            # Read the scenario matrix generated in scenarios_and_recipes.py
            matrix = pd.read_csv(scenario_matrix_path.format('pwc', region, year))

            # Create a 'metadata' table of scenario counts
            get_scenario_counts(matrix, sample_size).to_csv(pwc_scenario_path.format(region, year, 'meta'))

            # Randomly sample from each crop group and save the sample
            for crop_group, group_name in cultivated_crops:
                selection = sample_scenarios(matrix, crop_group, sample_size)[fields.fetch('pwc_scenario')]
                print(selection.weather_grid.unique())
                if not selection.empty:
                    selection.to_csv(pwc_scenario_path.format(region, year, group_name))
                    fetch_metfiles(selection.weather_grid.unique(), weather, pwc_metfile_path, overwrite_met)


main()
