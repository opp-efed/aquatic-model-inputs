import os
import numpy as np
import pandas as pd


class ScenarioMatrix(object):
    def __init__(self, mode, region, year, scenario_matrix_path, combinations, soil_data, crop_data, met_data):
        self.mode = mode
        self.aggregate = True if self.mode == 'sam_aggregated' else False

        # Set paths
        self.outfile = scenario_matrix_path.format(mode, region, year)
        self.qc_outfile = self.outfile.rstrip(".csv") + "_qa.csv"

        self.matrix = combinations

        self.add_double_crops(crop_data)

        self.matrix = self.matrix \
            .merge(soil_data, on="soil_id", how="left") \
            .merge(met_data, on="weather_grid", how="left") \
            .merge(crop_data, on=["cdl", "state"], how="left")

        # Perform other scenario attribution tasks
        print("\t\tPerforming scenario attribution...")
        self.scenario_attribution()

        # Clean up matrix and write to file
        print("\t\tPerforming QC check...")
        self.finalize()

        self.save()

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
        self.matrix.loc[self.matrix.root_zone_max < self.matrix.amxdr, 'amxdr'] = self.matrix.root_zone_max

    def add_double_crops(self, crop_data):
        self.matrix['overlay'] = 0
        self.matrix['original_cdl'] = self.matrix.cdl

        # Make new rows for double crops
        all_new = pd.DataFrame(columns=self.matrix.columns)
        for original_cdl, *new_classes in crop_data.double_crops.values:
            old_rows = self.matrix[self.matrix.cdl == original_cdl]
            for i, new_class in enumerate(new_classes):
                new_rows = old_rows.copy()
                new_rows.cdl = new_class
                new_rows.overlay = i
                all_new = all_new.append(new_rows)

        # Drop double cropped rows and add the new ones
        self.matrix = self.matrix[~np.in1d(self.matrix.cdl, crop_data.double_crops.cdl)]
        self.matrix = self.matrix.append(all_new)

    def finalize(self):
        """ Perform a quality check and fill missing data """
        from utilities import fields

        fields.refresh()
        # Add the appropriate fields
        if self.mode == 'pwc':
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

    def save(self):
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


class Combinations(object):
    def __init__(self, combo_path, region, aggregation_map, gridcodes):
        self.path = combo_path
        self.region = region
        self.map = aggregation_map
        self.gridcodes = gridcodes

        self.unique_fields = ['scenario_id', 'weather_grid', 'cdl', 'soil_id']
        self.matrix = None

    def read(self, year):
        # Unpack combinations table
        matrix = pd.DataFrame(dtype=np.int32, **np.load(self.path.format(self.region, year)))
        # Matrix headings are getting written as binary for some reason
        matrix = \
            matrix.rename(
                columns=dict(zip(matrix.columns.values, map(lambda x: x.decode('ascii'), matrix.columns.values))))

        # Perform aggregation, if needed
        if self.map is not None:
            matrix = matrix.merge(self.map, on='mukey', how='left')
            del matrix['mukey']
            matrix = matrix.groupby(['gridcode', 'weather_grid', 'cdl', 'aggregation_key']).sum().reset_index()
            matrix = matrix.rename(columns={'aggregation_key': 'soil_id'})
        else:
            matrix = matrix.rename(columns={'mukey': 'soil_id'})

        matrix = matrix.merge(self.gridcodes, how='left', on='gridcode')

        # Create a CDL/weather/soil identifier to link Recipes and Scenarios
        matrix['scenario_id'] = 's' + matrix.soil_id.astype("str") + \
                                'w' + matrix.weather_grid.astype("str") + \
                                'lc' + matrix.cdl.astype("str")

        return matrix

    def update(self, combos):
        # Remove catchment field
        combos = combos[['scenario_id', 'weather_grid', 'cdl', 'soil_id'] + ['area']]

        # Add new combos to existing table
        if self.matrix is None:
            self.matrix = combos
        else:
            self.matrix = pd.concat([self.matrix, combos])

        # Add up the areas of duplicate scenarios
        self.matrix = self.matrix.groupby(['scenario_id', 'weather_grid', 'cdl', 'soil_id']).sum().reset_index()


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
    sub_dir = "PWC" if mode == 'pwc' else 'SAM'
    soils = pd.read_csv(soil_path.format(sub_dir, region)).rename(
        columns={"mukey": "soil_id", "aggregation_key": 'soil_id'})
    if 'sam' in mode:
        aggregation = pd.read_csv(soil_path.format(sub_dir, region) + "_map.csv")
    else:
        aggregation = None
    return soils, aggregation


def main():
    from paths import \
        combo_path, condensed_nhd_path, met_attributes_path, processed_soil_path, recipe_path, scenario_matrix_path

    # Specify run parameters here
    regions = ['07']  # all regions: nhd_regions
    years = ['2010']
    generate_recipes = True
    scenario_modes = ['pwc']  # Select from 'pwc', 'sam_aggregated', or 'sam_unaggregated'

    # Read input tables not specific to regions
    met_data = read_met_data(met_attributes_path)
    crop_data = read_crop_data()

    for mode in scenario_modes:

        print("Creating {} scenarios...".format(mode.upper()))
        for region in regions:

            soil_data, aggregation_map = read_soils_data(region, processed_soil_path, mode)
            gridcodes = pd.DataFrame(**np.load(condensed_nhd_path.format(region)))[['gridcode', 'comid']]

            # Initialize table for all scenarios in all years
            all_combos = Combinations(combo_path, region, aggregation_map, gridcodes)

            for year in years:
                print("\tReading combinations and assigning Scenario ID for region {}, year {}...".format(region, year))
                # At this point, combos are unique combinations of soil, land use, weather, and catchment with areas
                combos = all_combos.read(year)
                if generate_recipes and 'sam' in mode:
                    print("\tGenerating recipes...")
                    Recipes(region, year, combos, recipe_path)
                all_combos.update(combos)

            print("\tGenerating scenarios for region {}, {}...".format(region, year))
            ScenarioMatrix(mode, region, year, scenario_matrix_path, all_combos.matrix, soil_data, crop_data, met_data)


main()
