import numpy as np
import os
from utilities import read_dbf, read_gdb
import pandas as pd


class CustomSSURGO(object):
    def __init__(self, ssurgo_path, output_path, overwrite):
        from utilities import fields
        from parameters import states

        self.in_folder = ssurgo_path
        self.out_folder = output_path.format()
        self.overwrite = overwrite
        self.fields = fields

        self._value_table = None

        # Initialize output folder if it doesn't exist
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

        # Extract each state
        for state in states:
            print(state)
            state_gdb = os.path.join(ssurgo_path, "gSSURGO_{}.gdb".format(state).lower())
            if os.path.exists(state_gdb):
                self.extract_tables(state_gdb, state)
            else:
                print("No SSURGO data found for {}".format(state))

    @property
    def value_table(self):
        if self._value_table is None:
            value_table = os.path.join(self.in_folder, "valu_fy2016.gdb")
            valu1 = read_gdb(value_table, "valu1", self.fields.fetch_old('valu1'))
            valu1 = valu1.rename(columns=self.fields.convert)
            self._value_table = valu1
        return self._value_table

    def extract_tables(self, gdb, state):

        out_file = os.path.join(self.out_folder, state)
        if self.overwrite or not all(map(os.path.exists, (out_file + '_horizon.csv', out_file + '_parameters.csv'))):
            print("Generating {}...".format(out_file))

            # Read horizon table
            chorizon_table = read_gdb(gdb, 'chorizon', self.fields.fetch_old('chorizon'))
            chorizon_table = chorizon_table.sort_values(['cokey', 'hzdept_r']).rename(columns=self.fields.convert)

            # Load other tables and join
            component_table = read_gdb(gdb, 'component', self.fields.fetch_old('component'))
            muaggatt_table = read_gdb(gdb, 'muaggatt', self.fields.fetch_old('muaggatt'))
            state_data = component_table.merge(muaggatt_table, on='mukey').merge(self.value_table, on='mukey')
            state_data = state_data.rename(columns=self.fields.convert)

            # Write to file
            chorizon_table.to_csv(out_file + '_horizon.csv')
            state_data.to_csv(out_file + '_params.csv')


class CustomNHDPlus(object):
    def __init__(self, nhd_path, out_path, overwrite):
        from parameters import nhd_regions

        self.nhd_path = nhd_path
        self.overwrite = overwrite

        for region in nhd_regions:
            outfile = out_path.format(region)
            if self.overwrite or not os.path.exists(outfile):
                print("Extracting tables for region {}...".format(region))
                extracted = self.extract_tables(region)
                self.save(extracted, outfile)

    def extract_tables(self, region):
        from utilities import fields
        from parameters import erom_months, vpus_nhd

        # Extract all files except EROM
        region_path = self.nhd_path.format(vpus_nhd[region], region)
        tables = \
            [(os.path.join("NHDPlusAttributes", "PlusFlowlineVAA.dbf"), "VAA"),
             (os.path.join("NHDPlusAttributes", "PlusFlow.dbf"), "PlusFlow"),
             (os.path.join("NHDSnapshot", "Hydrography", "NHDFlowline.dbf"), "Flowline"),
             (os.path.join("NHDPlusCatchment", "featureidgridcode.dbf"), "Gridcode")]
        erom_path = os.path.join(region_path, "EROMExtension", "EROM_{}0001.dbf")

        # Initialize output table
        hydro_table = None

        # Extract attribute tables
        for table, name in tables:
            print(table, name)
            new_table = read_dbf(os.path.join(region_path, table), fields.fetch_old(name))
            new_table = new_table[fields.fetch_old(name)].rename(columns=fields.convert)
            hydro_table = hydro_table.merge(new_table, on='comid',
                                            how='outer') if hydro_table is not None else new_table

        # Extract EROM
        for month in erom_months:
            print(month)
            erom_fields = {}
            for old, new in zip(fields.fetch_old('EROM'), fields.fetch('EROM')):
                if old != 'comid':
                    erom_fields[old] = "{}_{}".format(new, month)
            erom_table = read_dbf(erom_path.format(month))[fields.fetch_old('EROM')].rename(columns=erom_fields)
            hydro_table = hydro_table.merge(erom_table, on='comid', how='outer')

        # Expand field object to reflect new, monthly erom
        fields.expand('monthly')

        # Get rid of comid == 0.  These represent reaches upstream of a headwater (don't exist)
        hydro_table = hydro_table[hydro_table.comid != 0]

        return hydro_table[fields.fetch('condensed_nhd')].reset_index(drop=True)

    @staticmethod
    def save(extracted, outfile):
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        np.savez_compressed(outfile, data=extracted.values, columns=extracted.columns.values)


def main():
    from paths import soil_path, nhd_path, condensed_soil_path, condensed_nhd_path

    process_ssurgo = False
    process_nhd = True
    overwrite = True

    if process_ssurgo:
        CustomSSURGO(soil_path, condensed_soil_path, overwrite)

    if process_nhd:
        CustomNHDPlus(nhd_path, condensed_nhd_path, overwrite)


main()
