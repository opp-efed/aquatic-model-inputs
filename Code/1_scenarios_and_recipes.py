import modify
import numpy as np
import read


def create_recipes(scenarios):
    comids = scenarios.pop('comid')
    recipes = scenarios[['scenario_id', 'area']]

    # Get the first and last row corresponding to each comid
    first_rows = np.hstack(([0], np.where(comids[:-1] != comids[1:])[0] + 1))
    last_rows = np.hstack((first_rows[1:], [comids.size]))
    recipe_map = np.vstack((comids[first_rows], first_rows, last_rows)).T

    return recipes, recipe_map


def create_scenarios(combinations, met_data, crop_data, gridcodes, soil_data):
    # Join soils to scenarios. If aggregating soils, drop duplicates
    combined = combinations.merge(soil_data, how="left", on="mukey")
    if 'aggregation_key' in soil_data.columns:
        combined = combined.rename(columns={'aggregation_key': 'soil_id'})
        combined = combined.groupby(['gridcode', 'weather_grid', 'cdl', 'soil_id']).sum().reset_index()
        del combined['mukey']
    else:
        combined = combinations.rename(columns={'mukey': 'soil_id'})

    # Combine all input tables
    combined = combined.merge(met_data, how="left", on="weather_grid") \
        .merge(crop_data, how="left", on=["cdl", "state"]) \
        .merge(gridcodes, how='left', on="gridcode")

    return combined


def main():
    # Set run parameters
    regions = ['07']
    years = range(2010, 2015)
    aggregate = True
    depth_weight = False

    # Read data indexed to weather grid or crop
    met_data = read.met()
    crop_data = read.crop_data()
    for region in regions:
        # gridcodes = read.gridcodes(region)
        gridcodes = None

        # Read and process soils data
        soil_data = read.soils(region)
        soil_data = modify.process_soils(soil_data)
        if depth_weight:
            soil_data = modify.depth_weight_soils(soil_data)
        if aggregate:
            soil_data = modify.aggregate_soils(soil_data)

        for year in years:
            print("Reading combinations...")
            combinations = read.combinations(region, year)
            annual_scenarios = \
                create_scenarios(combinations, met_data, crop_data, gridcodes, soil_data)
            for year in years:
                recipes, recipe_map = create_recipes(annual_scenarios)


if __name__ == "__main__":
    main()
