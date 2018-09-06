import os
import numpy as np
import pandas as pd
from scipy import stats

from utilities import fields


class RegionSoils(object):
    def __init__(self, region, states, ssurgo, out_path):
        # Initialize variables
        self.region = region
        self.states = states
        self.ssurgo = ssurgo
        self.out_path = out_path

        # Refresh fields
        fields.refresh()

        # Read data from SSURGO
        print("\tReading raw soils data...")
        self.soil_table = self.read_tables()

        print("\tSelecting components...")
        self.select_components()

        print("\tAdjusting data...")
        self.adjust_data()

        print("\tExtending horizons...")
        self.sort_horizons()

        print("\tAdding soil attributes...")
        self.soil_attribution()

        print("\tWriting unaggregated soil table to file...")
        self.write_to_file()

        print("\tDepth weighting...")
        self.depth_weighting()

        print("\tWriting depth weighted soil table to file...")
        self.write_to_file('depth_weighted', perform_qc=False)

        print("\tAggregating soils...")
        aggregation_map = self.aggregate_soils()

        # Write to file
        print("\tWriting aggregated soil table to file...")
        self.write_to_file('aggregated', aggregation_map=aggregation_map)

    def adjust_data(self):
        self.soil_table.loc[:, 'orgC'] /= 1.724
        self.soil_table.loc[:, ['fc', 'wp']] /= 100.
        self.soil_table['thickness'] = self.soil_table['horizon_bottom'] - self.soil_table['horizon_top']

    def aggregate_soils(self):
        from parameters import bins

        # Sort data into bins
        out_data = [self.soil_table.hsg_letter]
        for field, field_bins in bins.items():
            labels = [field[:2 if field == "slope" else 1] + str(i) for i in range(1, len(field_bins))]
            sliced = pd.cut(self.soil_table[field].fillna(0), field_bins,
                            labels=labels, right=False, include_lowest=True)
            out_data.append(sliced.astype("str"))
        soil_agg = pd.concat(out_data, axis=1)

        # Add aggregation ID
        self.soil_table['aggregation_key'] = \
            soil_agg['hsg_letter'] + soil_agg['slope'] + soil_agg['orgC_5'] + soil_agg['sand_5'] + soil_agg['clay_5']
        aggregation_map = self.soil_table[['mukey', 'aggregation_key']]

        # Group by aggregation ID and take the mean of all properties except HSG, which will use mode
        grouped = self.soil_table.groupby('aggregation_key')
        averaged = grouped.mean().reset_index()
        hydro_group = grouped['hydro_group'].agg(lambda x: stats.mode(x)[0][0]).to_frame().reset_index()
        del averaged['hydro_group']
        self.soil_table = averaged.merge(hydro_group, on='aggregation_key')

        return aggregation_map

    def depth_weighting(self):
        from parameters import depth_bins, max_horizons

        # Get the root name of depth weighted fields
        depth_fields = {f.split("_")[0] for f in fields.fetch('depth_weight')}
        depth_weighted = []

        # Generate weighted columns for each bin
        for bin_top, bin_bottom in zip([0] + list(depth_bins[:-1]), list(depth_bins)):
            bin_table = np.zeros((self.soil_table.shape[0], len(depth_fields)))

            # Perform depth weighting on each horizon
            for i in range(max_horizons):
                # Set field names for horizon
                top_field, bottom_field = 'horizon_top_{}'.format(i + 1), 'horizon_bottom_{}'.format(i + 1)
                value_fields = ["{}_{}".format(f, i + 1) for f in depth_fields]

                # Adjust values by bin
                horizon_bottom, horizon_top = self.soil_table[bottom_field], self.soil_table[top_field]
                overlap = (horizon_bottom.clip(upper=bin_bottom) - horizon_top.clip(lower=bin_top)).clip(0)
                ratio = (overlap / (horizon_bottom - horizon_top)).fillna(0)
                bin_table += self.soil_table[value_fields].fillna(0).mul(ratio, axis=0).values

            # Add columns
            bin_table = \
                pd.DataFrame(bin_table, columns=["{}_{}".format(f, bin_bottom) for f in depth_fields])
            depth_weighted.append(bin_table)

        # Clear old fields and append new one
        for field in fields.fetch('horizons_expanded'):
            del self.soil_table[field]
        self.soil_table = pd.concat([self.soil_table] + depth_weighted, axis=1)

    def read_tables(self):

        horizon_tables, component_tables = [], []
        for state in self.states:
            horizons, components = self.ssurgo.fetch(state)
            if components is not None:
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

    def sort_horizons(self):
        from parameters import max_horizons

        # Get number of horizons for each map unit
        grouped = self.soil_table.groupby('cokey')
        horizon_counts = grouped.cumcount() + 1  # horizon count
        self.soil_table = \
            self.soil_table.merge(grouped.size().reset_index(name="n_horizons"), on='cokey')

        # Get fields to be extended by horizon
        horizontal_fields = fields.fetch('horizontal') + ['kwfact']

        # Extend columns of data for multiple horizons
        horizontal_data = \
            self.soil_table[['cokey'] + horizontal_fields] \
                .set_index(['cokey', horizon_counts]).unstack().sort_index(1, level=1)

        horizontal_data.columns = ['_'.join(map(str, i)) for i in horizontal_data.columns]
        horizontal_data = horizontal_data.reset_index()

        # Add dummy fields (TODO - better way than this?)
        for field in fields.fetch('horizontal'):
            for i in range(horizon_counts.max(), max_horizons):
                horizontal_data["{}_{}".format(field, i + 1)] = np.nan

        # Fold horizontal data back into soil table
        print(self.soil_table.columns.values)
        keep_fields = [f for f in self.soil_table.columns if f not in horizontal_fields]
        print(keep_fields)
        self.soil_table = \
            self.soil_table[keep_fields].drop_duplicates().merge(horizontal_data, on='cokey')

    def select_components(self):
        """  Identify component to be used for each map unit """
        # Isolate unique map unit/component pairs
        components = self.soil_table[['mukey', 'cokey', 'major_component', 'component_pct']].drop_duplicates(
            ['mukey', 'cokey'])

        # Select major compoments
        components = components[components.major_component == 'Yes']

        # Select major component with largest area (comppct)
        components = components.sort_values('component_pct', ascending=False)
        components = components[~components.mukey.duplicated()]
        self.soil_table = components.merge(self.soil_table, on=['mukey', 'cokey'], how='left')

    def soil_attribution(self):
        """ Merge soil table with params and get hydrologic soil group (hsg) and USLE values """

        def calculate_uslels(df):
            from utilities import uslels_matrix

            row = (uslels_matrix.index.values.astype(np.int32) < df.slope).sum()
            col = (uslels_matrix.columns.values.astype(np.int32) < df.slope_length).sum()
            try:
                return uslels_matrix.iloc[row, col]
            except IndexError:
                return np.nan

        from parameters import hydro_soil_groups, uslep_values, bins

        # New HSG code - take 'max' of two versions of hsg
        hsg_to_num = {hsg: i + 1 for i, hsg in enumerate(hydro_soil_groups)}
        num_to_hsg = \
            dict(zip(hsg_to_num.values(), map(lambda x: x.replace("/", ""), hsg_to_num.keys())))
        self.soil_table['hydro_group'] = \
            self.soil_table[['hydro_group', 'hydro_group_dominant']].applymap(
                lambda x: hsg_to_num.get(x)).max(axis=1).fillna(-1).astype(np.int32)
        self.soil_table['hsg_letter'] = self.soil_table['hydro_group'].map(num_to_hsg)

        # Calculate USLE LS and P values
        self.soil_table['uslels'] = self.soil_table.apply(calculate_uslels, axis=1)
        self.soil_table['uslep'] = \
            np.array(uslep_values)[np.int16(pd.cut(self.soil_table.slope, bins['slope'], labels=False))]

        # Select kwfact
        self.soil_table['kwfact'] = self.soil_table.kwfact_1
        self.soil_table.loc[self.soil_table.kwfact == 9999, 'kwfact'] = self.soil_table.kwfact_2
        self.soil_table.loc[self.soil_table.kwfact == 9999, 'kwfact'] = np.nan
        self.soil_table.loc[(self.soil_table.horizon_top_1 > 1) | (self.soil_table.horizon_top_1 < 0), 'kwfact'] = \
            np.nan
        self.soil_table.loc[self.soil_table.desgnmaster_1 == 'R', 'kwfact'] = np.nan
        for field in self.soil_table.columns.values:
            if 'kwfact_' in field:
                del self.soil_table[field]

    def write_to_file(self, modify=None, aggregation_map=None, perform_qc=True):

        # Reset output fields
        fields.refresh()

        # Select output parameters based on output type
        if modify == 'aggregated':
            fields.expand('depth')
            out_fields = fields.fetch("sam_ssurgo") + ['aggregation_key']
            mode = "sam_aggregated"
        elif modify == 'depth_weighted':
            fields.expand('depth')
            out_fields = fields.fetch("sam_ssurgo") + ['mukey']
            mode = "sam_unaggregated"
        elif modify is not None:
            raise Exception("Invalid output type {} given, must be 'aggregated', 'depth_weighted', or None")
        else:
            fields.expand('horizon')
            mode = "PWC"
            out_fields = fields.fetch("pwc_ssurgo")

        out_path = self.out_path.format(mode, self.region)

        # Create output folder if it doesn't exist
        if not os.path.isdir(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        # Confine to output fields
        out_table = self.soil_table[out_fields]

        # Perform QC check
        if perform_qc:
            fields.perform_qc(out_table, out_path + "_qc.csv")

        # Write table
        out_table.reset_index(drop=True).to_csv(out_path + '.csv', index=False, float_format='%3.2f')
        if modify == 'aggregated':
            aggregation_map.to_csv(out_path + '_map.csv', index=False)


class SSURGOReader(object):
    def __init__(self, ssurgo_dir):
        self.path = ssurgo_dir

    def fetch(self, state):
        horizon_table = os.path.join(self.path, '{}_horizon.csv'.format(state.upper()))
        component_table = os.path.join(self.path, '{}_params.csv'.format(state.upper()))
        if all(map(os.path.exists, (horizon_table, component_table))):
            return pd.read_csv(horizon_table, index_col=0), pd.read_csv(component_table, index_col=0)
        else:
            print("Processed soils not found for {}".format(state))
            return None, None


def main():
    from parameters import states_nhd
    from paths import condensed_soil_path, processed_soil_path

    # Initialize SSURGO reader
    ssurgo = SSURGOReader(condensed_soil_path)
    regions = states_nhd.keys()

    # Iterate through states
    for region in regions:
        print("Processing Region {} soils...".format(region))
        states = states_nhd[region]
        RegionSoils(region, states, ssurgo, processed_soil_path)


main()
