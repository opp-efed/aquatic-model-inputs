import os
import pandas as pd
from paths import sam_scenario_path, pwc_scenario_path


def create_dir(out_path):
    """ Create a directory for a file name if it doesn't exist """
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))


def scenarios(scenario_matrix, mode, region, name=None):
    if mode == 'sam':
        out_path = sam_scenario_path.format(region)
    elif mode == 'pwc':
        out_path = pwc_scenario_path.format(region, name)
    create_dir(out_path)
    scenario_matrix.to_csv(out_path, index=False)


def recipes(recipe_matrix, recipe_map):
    pass
