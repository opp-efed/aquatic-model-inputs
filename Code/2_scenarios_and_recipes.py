import os
import re
import numpy as np
import pandas as pd


class ScenarioMatrix(object):
    def __init__(self, region, year, mode, scenarios, soil_data, crop_params, crop_dates, met_data, output_format):

        self.mode = mode
        self.aggregate = True if self.mode == 'sam_aggregated' else False

        # Set paths
        self.outfile = output_format.format(mode, region, year)
        self.qc_outfile = self.outfile.rstrip(".csv") + "_qa.csv"

        # Create duplicate scenarios for double-cropped classes
        self.add_double_crops()

        # Merge all tables
        print("\t\tMerging tables...")
        self.matrix = scenarios.merge(crop_params, on='cdl', how='left')
        self.matrix = self.matrix.merge(crop_dates, on=('weather_grid', 'gen_class'), how='left')
        self.matrix = self.matrix.merge(soil_data, on="soil_id", how="left")
        self.matrix = self.matrix.merge(met_data, left_on='weather_grid', right_on='stationID', how='left')

        # Perform other scenario attribution tasks
        print("\t\tPerforming scenario attribution...")
        self.scenario_attribution()

        # Clean up matrix and write to file
        print("\t\tPerforming QC check...")
        self.finalize()

        self.save()

    def scenario_attribution(self):

        # Process curve number
        from parameters import hydro_soil_groups, cultivated_crops

        # Determine which land cover classes are cultivated
        self.matrix['cultivated'] = self.matrix.cdl.isin(cultivated_crops)

        # Assign curve numbers for cultivated classes
        num_to_hsg = dict(enumerate(hydro_soil_groups))
        num_to_hsg.update({2: "A", 4: "B", 6: "C"})  # A/D -> A, B/D -> B, C/D -> C
        for num, hsg in num_to_hsg.items():
            selected_rows = (self.matrix.hydro_group == num) & (self.matrix.cultivated == 1)
            self.matrix.loc[selected_rows, 'cn_ag'] = self.matrix['cn_ag_' + hsg]
            self.matrix.loc[selected_rows, 'cn_fallow'] = self.matrix['cn_fallow_' + hsg]

        # Assign curve numbers for non-cultivated classes
        num_to_hsg.update({2: "D", 4: "D", 6: "D"})  # A/D -> A, B/D -> B, C/D -> C
        for num, hsg in num_to_hsg.items():
            selected_rows = (self.matrix.hydro_group == num) & (self.matrix.cultivated == 0)
            self.matrix.loc[selected_rows, 'cn_ag'] = self.matrix['cn_ag_' + hsg]
            self.matrix.loc[selected_rows, 'cn_fallow'] = self.matrix['cn_fallow_' + hsg]

        # Deal with maximum rooting depth
        self.matrix.loc[self.matrix.root_zone_max < self.matrix.amxdr, 'amxdr'] = self.matrix.root_zone_max

        # Assign snowmelt factor (sfac)
        self.matrix['sfac'] = 0.36
        self.matrix.loc[self.matrix.cdl.isin((60, 70, 140, 190)), 'sfac'] = .16

        # Assign missing crop dates
        empty = pd.isnull(self.matrix[['plant_begin', 'emergence_begin', 'maxcover_begin', 'harvest_begin']])
        emergence = ~empty.plant_begin & empty.emergence_begin
        max_cover = ~empty.plant_begin & ~empty.harvest_begin & empty.maxcover_begin

        # Emergence 7 days after planting, maturity halfway between plant and harvest
        self.matrix.loc[emergence, 'emergence_begin'] = self.matrix.plant_begin[emergence] + 7
        self.matrix.loc[max_cover, 'maxcover_begin'] = \
            (self.matrix.plant_begin[max_cover] + self.matrix.harvest_begin[max_cover]) / 2

    def add_double_crops(self):
        """ Join CDL-class-specific parameters to the table and add new rows for double cropped classes """
        from parameters import double_crops

        # Process double crops
        self.matrix['orig_cdl'] = self.matrix['cdl']
        self.matrix['overlay'] = 0  # Overlay crops aren't used to generate runoff in pesticide calculator
        all_new = []
        for old_crop, new_crops in double_crops.items():
            for i, new_crop in enumerate(new_crops):
                new_rows = self.matrix[self.matrix.orig_cdl == old_crop].copy()
                new_rows['cdl'] = new_crop
                new_rows['overlay'] = i
                all_new.append(new_rows)
        new_data = pd.concat(all_new, axis=0)
        self.matrix = pd.concat([self.matrix, new_data], axis=0)

    def finalize(self):
        """ Perform a quality check and fill missing data """
        from utilities import fields

        fields.refresh()
        # Add the appropriate fields
        if self.mode == 'pwc':
            # Count the number of horizons in
            max_horizons = \
                max([int(m.group(1)) for m in (re.match(".+?_(\d+)$", f) for f in self.matrix.columns.values) if m])
            fields.expand('horizon', max_horizons)
            out_fields = fields.fetch('pwc_scenario')
        else:
            fields.expand('depth')
            out_fields = fields.fetch('sam_scenario')

        # Rename soil id back to mukey if not aggregating
        if not self.aggregate:
            self.matrix = self.matrix.rename(columns={'soil_id': 'mukey'})

        # Trim to output fields
        self.matrix = self.matrix[out_fields].reset_index(drop=True)

        # Perform QC
        fields.perform_qc(self.matrix, self.qc_outfile)

        # Fill missing data
        self.matrix.fillna(fields.fill_value, inplace=True)

    def save(self):
        self.matrix.to_csv(self.outfile, index=False)


class Recipes(object):
    def __init__(self, region, year, combos, output_format):
        print(combos.columns)
        self.outfile = output_format.format(region, year)
        self.comids = combos.comid.as_matrix()
        self.recipes = combos[['scenario_id', 'area']]

        # Generate a map of which rows correspond to which comids
        self.recipe_map = self.map_recipes()

        # Save to file
        self.save()

    def map_recipes(self):
        """ Get the first and last row corresponding to each comid  """

        first_rows = np.hstack(([0], np.where(self.comids[:-1] != self.comids[1:])[0] + 1))
        last_rows = np.hstack((first_rows[1:], [self.comids.size]))
        return np.vstack((self.comids[first_rows], first_rows, last_rows)).T

    def save(self):
        if not os.path.exists(os.path.dirname(self.outfile)):
            os.makedirs(os.path.dirname(self.outfile))
        np.savez_compressed(self.outfile, data=self.recipes.as_matrix(), map=self.recipe_map)


def read_tables(crop_params_path, crop_dates_path, curve_numbers_path, metfile_path):
    from utilities import fields

    # Read and modify crop dates
    crop_dates = pd.read_csv(crop_dates_path)
    crop_dates = crop_dates[fields.fetch_old('CropDates')].rename(columns=fields.convert)
    crop_dates['emergence_begin'], crop_dates['emergence_end'] = 0, 0  # jch - temporary

    # Read crop params
    crop_params = pd.read_csv(crop_params_path)[fields.fetch_old('CropParams')].rename(columns=fields.convert)
    curve_numbers = pd.read_csv(curve_numbers_path)[fields.fetch_old('CurveNumbers')].rename(columns=fields.convert)
    crop_params = crop_params.merge(curve_numbers, on='gen_class', how='left')

    # Read table with weather  grid parameters
    met_data = pd.read_csv(metfile_path)

    return crop_params, crop_dates, met_data


def read_combinations(region, year, combo_path, aggregation_map, nhd_path):
    from parameters import vpus_nhd

    # Unpack combinations table
    matrix = pd.DataFrame(dtype=np.int32, **np.load(os.path.join(combo_path, "{}_{}.npz".format(region, year))))

    # Perform aggregation, if needed
    if aggregation_map is not None:
        matrix = matrix.merge(aggregation_map, on='mukey', how='left')
        del matrix['mukey']
        matrix = matrix.groupby(['gridcode', 'weather_grid', 'cdl', 'aggregation_key']).sum().reset_index()
        matrix = matrix.rename(columns={'aggregation_key': 'soil_id'})
    else:
        matrix = matrix.rename(columns={'mukey': 'soil_id'})

    # Convert gridcode to comid
    nhd_table = pd.DataFrame(**np.load(nhd_path.format(vpus_nhd[region], region)))[['gridcode', 'comid']]
    matrix = matrix.merge(nhd_table, how='left', on='gridcode')
    del matrix['gridcode']

    # Create a CDL/weather/soil identifier to link Recipes and Scenarios
    matrix['scenario_id'] = 's' + matrix.soil_id.astype("str") + \
                            'w' + matrix.weather_grid.astype("str") + \
                            'lc' + matrix.cdl.astype("str")

    return matrix


def read_soils(region, soil_path, mode):
    sub_dir = "PWC" if mode == 'pwc' else 'SAM'
    soils = pd.read_csv(soil_path.format(sub_dir, region)).rename(
        columns={"mukey": "soil_id", "aggregation_key": 'soil_id'})
    if 'sam' in mode:
        aggregation = pd.read_csv(soil_path.format(sub_dir, region) + "_map.csv")
    else:
        aggregation = None
    return soils, aggregation


def main():
    from parameters import states_nhd

    from paths import combo_path, condensed_nhd_path, processed_ssurgo_path, \
        crop_params_path, curve_numbers_path, crop_dates_path, met_attributes_path, recipe_path, scenario_matrix_path

    # Specify run parameters here
    regions = ['07']  # all regions: sorted(states_nhd.keys())
    years = ['2010']
    generate_recipes = True
    scenario_modes = ['sam_aggregated', 'pwc']  # Select from 'pwc', 'sam_aggregated', or 'sam_unaggregated'

    # Read input tables not specific to regions
    print("Reading tables...")
    crop_params, crop_dates, met_data = \
        read_tables(crop_params_path, crop_dates_path, curve_numbers_path, metfile_path)

    for mode in scenario_modes:

        print("Running {} mode...".format(mode))

        for region in regions:

            print("\tReading soil data...")
            regional_soil, aggregation_map = read_soils(region, processed_ssurgo_path, mode)

            # Initialize table for all scenarios in all years
            scenarios = None

            for year in years:

                print("\tReading combinations and assigning Scenario ID for region {}, year {}...".format(region, year))
                combos = read_combinations(region, year, combo_path, aggregation_map, condensed_nhd_path)

                if generate_recipes and 'sam' in mode:
                    print("\tGenerating recipes...")
                    Recipes(region, year, combos, recipe_path)

                combos = combos[['weather_grid', 'cdl', 'soil_id', 'scenario_id']]
                if scenarios is None:
                    scenarios = combos.drop_duplicates()
                else:
                    scenarios = pd.concat([scenarios, combos]).drop_duplicates()

            print("\tGenerating scenarios for region {}, {}...".format(region, year))
            ScenarioMatrix(region, year, mode, scenarios, regional_soil, crop_params, crop_dates, met_data,
                           scenario_matrix_path)
    exit()


main()
