import os
import numpy as np
import pandas as pd

from utilities import read_dbf


class Scenarios(object):
    def __init__(self, mode, region, scenario_matrix_path, combo_path):
        self.region = region
        self.aggregate = 'aggregated' in mode
        self.depth_weight = 'sam' in mode
        self.combo_path = combo_path

        # Set paths
        self.outfile = scenario_matrix_path.format(mode, region)
        self.qc_outfile = self.outfile.rstrip(".csv") + "_qa.csv"

        # Initialize the combination matrix
        self.matrix = None

    def add_double_crops(self, crop_data):
        self.matrix['overlay'] = 0
        self.matrix['original_cdl'] = self.matrix.cdl

        # Make new rows for double crops
        all_new = pd.DataFrame(columns=self.matrix.columns)
        for original_cdl, *new_classes in crop_data.double_crops.values:
            old_rows = self.matrix[self.matrix.cdl == original_cdl]
            for overlay, new_class in enumerate(new_classes):  # second crop is overlay
                new_rows = old_rows.copy()
                new_rows.cdl = new_class
                new_rows.overlay = overlay
                all_new = all_new.append(new_rows)

        # Drop double cropped rows and add the new ones
        self.matrix = self.matrix[~np.in1d(self.matrix.cdl, crop_data.double_crops.cdl)].append(all_new)

    def populate(self, soil_data, met_data, crop_data):
        """ Join combinations with corresponding data tables and perform additional attribution """

        # Join all tabular data to scenario matrix
        self.matrix = self.matrix \
            .merge(soil_data, on="soil_id", how="left") \
            .merge(met_data, on="weather_grid", how="left") \
            .merge(crop_data, on=["cdl", "state"], how="left")

        # Create duplicate scenarios for double crops
        self.add_double_crops(crop_data)

        # Perform other scenario attribution tasks
        print("\t\tPerforming scenario attribution...")
        self.scenario_attribution()

    def read_combinations(self, year, gridcodes, aggregation_map, cell_size):
        # Unpack combinations table
        from parameters import states_nhd

        matrix = None
        for state in states_nhd[self.region]:
            combo_file = self.combo_path.format(self.region, state, year)
            combo_table = pd.DataFrame(dtype=np.int32, **np.load(combo_file))
            matrix = combo_table if matrix is None else pd.concat([matrix, combo_table], axis=0)

        # Column names get written as binary by spatial_overlay
        matrix.columns = [f.decode('ascii') for f in matrix.columns.values]

        # Combine duplicate rows and convert counts to areas
        matrix = matrix.groupby(['gridcode', 'weather_grid', 'cdl', 'mukey']).sum().reset_index()
        matrix['area'] *= cell_size ** 2

        # Perform aggregation, if needed
        if self.aggregate:
            matrix = matrix.merge(aggregation_map, on='mukey', how='left')
            del matrix['mukey']
            matrix = matrix.rename(columns={'aggregation_key': 'soil_id'}) \
                .groupby(['gridcode', 'weather_grid', 'cdl', 'soil_id']).sum().reset_index()
        else:
            matrix = matrix.rename(columns={'mukey': 'soil_id'})

        matrix = matrix.merge(gridcodes, how='left', on='gridcode').rename(columns={"featureid": "comid"})
        del matrix['gridcode']

        # Create a CDL/weather/soil identifier to link Recipes and Scenarios
        matrix['scenario_id'] = 's' + matrix.soil_id.astype("str") + \
                                'w' + matrix.weather_grid.astype("str") + \
                                'lc' + matrix.cdl.astype("str")

        return matrix

    def update_combinations(self, combinations):
        # Remove catchment field
        combinations = combinations[['scenario_id', 'weather_grid', 'cdl', 'soil_id'] + ['area']]

        # Add new combos to existing table
        if self.matrix is None:
            self.matrix = combinations
        else:
            self.matrix = pd.concat([self.matrix, combinations])

        # Add up the areas of duplicate scenarios
        self.matrix = self.matrix.groupby(['scenario_id', 'weather_grid', 'cdl', 'soil_id']).sum().reset_index()

    def scenario_attribution(self):
        from parameters import hsg_cultivated, hsg_non_cultivated, null_curve_number
        from utilities import fields

        # Crop dates for double-cropped classes
        for field in fields.fetch("CropDates"):
            self.matrix[field] = self.matrix.pop(field + "_a")
            self.matrix.loc[self.matrix.overlay == 1, field] = \
                self.matrix.loc[self.matrix.overlay == 1, field + "_b"]
            del self.matrix[field + "_b"]

        # Emergence 7 days after planting, maturity halfway between plant and harvest
        self.matrix['emergence_begin'] = self.matrix.plant_begin + 7
        self.matrix['maxcover_begin'] = (self.matrix.plant_begin + self.matrix.harvest_begin) / 2

        # Process curve number
        self.matrix['cn_ag'] = self.matrix['cn_fallow'] = null_curve_number
        for hydro_soil_groups in hsg_cultivated, hsg_non_cultivated:
            for i, hsg in enumerate(hydro_soil_groups):
                sel = (self.matrix.hydro_group == i + 1) & (self.matrix.cultivated_cdl == 1)
                self.matrix.loc[sel, 'cn_ag'] = self.matrix.loc[sel, 'cn_ag_' + hsg]
                self.matrix.loc[sel, 'cn_fallow'] = self.matrix.loc[sel, 'cn_fallow_' + hsg]

        # Deal with maximum rooting depth
        self.matrix['amxdr'] = self.matrix[['amxdr', 'root_zone_max']].min(axis=1)

    def write_file(self):
        """ Perform a quality check and fill missing data """

        from utilities import fields

        fields.refresh()
        # Add the appropriate fields
        if not self.depth_weight:
            # Count the number of horizons in
            fields.expand('horizon')
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
        # fields.perform_qc(self.matrix, self.qc_outfile)

        # Fill missing data
        self.matrix.fillna(fields.fill_value, inplace=True)

        self.matrix.to_csv(self.outfile, index=False)


class Recipes(object):
    def __init__(self, region, year, combos, output_format):
        self.outfile = output_format.format(region, year)
        self.comids = combos.comid.values
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
        np.savez_compressed(self.outfile, data=self.recipes.values, map=self.recipe_map)


def read_gridcodes(nhd_path):
    gridcodes_path = os.path.join(nhd_path, "NHDPlusCatchment", "featureidgridcode.dbf")
    return read_dbf(gridcodes_path)[['featureid', 'gridcode']]


def read_met_data(met_attributes_path):
    met_data = pd.read_csv(met_attributes_path)
    del met_data['state']  # for now, use state parameter from SSURGO
    return met_data


def read_crop_data():
    """ Merge all parameter tables linked to crop class """

    # Since crop data table is read in more than 1 script, reading this table is done in utilities.CropMatrix
    from utilities import fields, crops as crop_data

    # Parse crop dates
    for field_stem in fields.fetch("CropDates"):
        if "plant" in field_stem:
            for var in "ab":
                field = field_stem + "_" + var
                harvest_field = field.replace("plant", "harvest")
                crop_data[field] = \
                    (pd.to_datetime(crop_data[field], format="%d-%b") - pd.to_datetime("1900-01-01")).dt.days
                crop_data[harvest_field] = \
                    (pd.to_datetime(crop_data[harvest_field], format="%d-%b") - pd.to_datetime("1900-01-01")).dt.days
                crop_data[crop_data[harvest_field] < crop_data[field]] += 365
    return crop_data


def read_soils_data(region, soil_path, mode):
    soils = pd.read_csv(soil_path.format(mode, region) + ".csv").rename(
        columns={"mukey": "soil_id", "aggregation_key": 'soil_id'})
    if 'sam' in mode:
        aggregation = pd.read_csv(soil_path.format(mode, region) + "_map.csv")
    else:
        aggregation = None
    return soils, aggregation


def main():
    from paths import \
        nhd_path, combo_path, met_attributes_path, processed_soil_path, recipe_path, scenario_matrix_path, scratch_dir
    from parameters import vpus_nhd, states_nhd

    # Specify run parameters here
    region = '07'  # all regions: nhd_regions
    years = range(2013, 2018)
    generate_recipes = True
    scenario_modes = ['pwc']  # Select from 'pwc', 'sam_aggregated', or 'sam_unaggregated'
    cell_size = 30

    # Read input tables not specific to regions
    met_data = read_met_data(met_attributes_path)
    crop_data = read_crop_data()

    for mode in scenario_modes:
        print("Creating {} scenarios...".format(mode.lower()))

        soil_data, aggregation_map = read_soils_data(region, processed_soil_path, mode)

        gridcodes = read_gridcodes(nhd_path.format(vpus_nhd[region], region))

        # Initialize table for all scenarios in all years
        scenarios = Scenarios(mode, region, scenario_matrix_path, combo_path)

        # Read all combination tables and generate recipes
        for year in years:
            scratch_path = os.path.join(scratch_dir, "scenarios{}{}".format(mode, year))
            print("\tReading combinations and assigning Scenario ID for region {}, year {}...".format(region, year))
            if os.path.exists(scratch_path):
                print(1)
                if year == 2017:
                    scenarios.matrix = pd.read_csv(scratch_path, index_col=0)
            else:
                print(2)
                combos = scenarios.read_combinations(year, gridcodes, aggregation_map, cell_size)
                if generate_recipes and 'sam' in mode:
                    print("\tGenerating recipes...")
                    Recipes(region, year, combos, recipe_path)
                scenarios.update_combinations(combos)
                scenarios.matrix.to_csv(scratch_path)

    matrix = scenarios.matrix.copy()
    chunk = 500000
    for i in range(0, matrix.shape[0], chunk):
        print(i)
        # Append data tables to scenarios and write to file
        scenarios.matrix = matrix[i:i + chunk]
        scenarios.outfile = scenarios.outfile + "_{}".format(i)
        scenarios.populate(soil_data, met_data, crop_data)
        scenarios.write_file()


main()
