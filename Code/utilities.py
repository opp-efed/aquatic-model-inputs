import os
import numpy as np
import datetime as dt
import pandas as pd

from paths import uslels_path, fields_and_qc_path

""" Functions and classes utilized by multiple scripts """


class FieldMatrix(object):
    def __init__(self):
        self.path = fields_and_qc_path
        self.refresh()

        self.extended_monthly = False
        self.depth_weighted = False
        self.horizons_expanded = False
        self._qc_table = None
        self._convert = None

    def data_type(self, fields=None, old_fields=False):
        if fields is None:
            data_types = self.matrix.data_type
        else:
            index_col = "internal_name" if not old_fields else "external_name"
            data_types = self.matrix.set_index(index_col).loc[fields].data_type.values
        return np.array(list(map(eval, data_types)))

    def expand(self, mode='depth'):

        from parameters import depth_bins, erom_months, max_horizons
        try:
            condition, select_field, numbers = \
                {'depth': ('depth_weighted', 'depth_weight', depth_bins),
                 'horizon': ('horizons_expanded', 'horizontal', range(1, max_horizons + 1)),
                 'monthly': ('extended_monthly', 'monthly', erom_months)}[mode]

        except KeyError as e:
            message = "Invalid expansion mode '{}' specified: must be in ('depth', 'horizon', 'monthly')".format(mode)
            raise Exception(message) from e

        # Test to make sure it hasn't already been done
        if not getattr(self, condition):

            # Find each row that applies, duplicate, and append to the matrix
            new_rows = []
            for _, row in self.matrix[self.matrix[select_field] == 1].iterrows():
                for i in numbers:
                    new_row = row.copy()
                    new_row['internal_name'] = row.internal_name + "_" + str(i)
                    new_row[condition] = 1
                    new_rows.append(new_row)

            # Filter out the old rows and add the new ones
            self.matrix = self.matrix[~self.matrix.internal_name.isin(self.fetch(select_field))]
            self.matrix = pd.concat([self.matrix, pd.concat(new_rows, axis=1).T], axis=0)

            # Record that the duplication has occurred
            setattr(self, condition, True)

    def fetch_field(self, item, field):
        for column in 'data_source', 'source_table':
            if item in self.matrix[column].values:
                return self.matrix[self.matrix[column] == item][field].tolist()
        if item in self.matrix.columns:
            selected = self.matrix[self.matrix[item] > 0]
            if selected[item].max() > 1:
                selected = selected.sort_values(item)
            return selected[field].tolist()
        else:
            print("Unrecognized sub-table '{}'".format(item))

    def fetch_old(self, item):
        return self.fetch_field(item, 'external_name')

    def fetch(self, item):
        return self.fetch_field(item, 'internal_name')

    @property
    def convert(self):
        if self._convert is None:
            self._convert = {row.external_name: row.internal_name for _, row in self.matrix.iterrows()}
        return self._convert

    @property
    def qc_table(self):
        if self._qc_table is None:
            qc_fields = ['range_min', 'range_max', 'range_flag',
                         'general_min', 'general_max', 'general_flag',
                         'blank_flag', 'fill_value']
            self._qc_table = self.matrix.set_index('internal_name')[qc_fields].dropna(subset=qc_fields, how='all')
        return self._qc_table

    def perform_qc(self, other, outfile=None, verbose=False):

        # Confine QC table to fields in other table
        missing_fields = set(other.columns.values) - set(self.qc_table.index.values)
        if verbose and any(missing_fields):
            print("Field(s) {} not found in QC table".format(", ".join(map(str, missing_fields))))
        active_fields = [field for field in self.qc_table.index.values if field in other.columns.tolist()]
        qc_table = self.qc_table.loc[active_fields]

        # Flag missing data
        flags = pd.isnull(other)[qc_table.index.values].mul(qc_table.blank_flag)

        # Flag out-of-range data
        for test in ('general', 'range'):
            ranges = qc_table[[test + "_min", test + "_max", test + "_flag"]].dropna()
            for param, (param_min, param_max, flag) in ranges.iterrows():
                out_of_range = (other[param] < param_min) | (other[param] > param_max)
                if out_of_range.any():
                    flags.loc[out_of_range, param] = np.maximum(flags.loc[out_of_range, param].values, flag)

        # Write QC file
        if outfile is not None:
            if not os.path.isdir(os.path.dirname(outfile)):
                os.makedirs(os.path.dirname(outfile))
            flags.to_csv(outfile)

        return flags

    @property
    def fill_value(self):
        return self.matrix.set_index('internal_name').fill_value.dropna()

    def refresh(self):
        # Read the fields/QC matrix
        if self.path is not None:
            self.matrix = pd.read_csv(self.path)
        elif self.matrix is not None:
            self.matrix = self.matrix

        self.erom_expanded = self.depth_weighted = self.horizons_expanded = False


class Navigator(object):
    def __init__(self, region_id, upstream_path):
        self.file = upstream_path.format(region_id, 'nav')
        self.paths, self.times, self.map, self.alias_to_reach, self.reach_to_alias = self.load()
        self.reach_ids = set(self.reach_to_alias.keys())

    def load(self):
        assert os.path.isfile(self.file), "Upstream file {} not found".format(self.file)
        data = np.load(self.file, mmap_mode='r')
        conversion_array = data['alias_index']
        reverse_conversion = dict(zip(conversion_array, np.arange(conversion_array.size)))
        return data['paths'], data['time'], data['path_map'], conversion_array, reverse_conversion

    def upstream_watershed(self, reach_id, mode='reach', return_times=False, return_warning=False, verbose=False):

        def unpack(array):
            first_row = [array[start_row][start_col:]]
            remaining_rows = list(array[start_row + 1:end_row])
            return np.concatenate(first_row + remaining_rows)

        # Look up reach ID and fetch address from pstream object
        reach = reach_id if mode == 'alias' else self.reach_to_alias.get(reach_id)
        reaches, adjusted_times, warning = np.array([]), np.array([]), None
        try:
            start_row, end_row, col = map(int, self.map[reach])
            start_col = list(self.paths[start_row]).index(reach)
        except TypeError:
            warning = "Reach {} not found in region".format(reach)
        except ValueError:
            warning = "{} not in upstream lookup".format(reach)
        else:
            # Fetch upstream reaches and times
            aliases = unpack(self.paths)
            reaches = aliases if mode == 'alias' else np.int32(self.alias_to_reach[aliases])

        # Determine which output to deliver
        output = [reaches]
        if return_times:
            times = unpack(self.times)
            adjusted_times = np.int32(times - self.times[start_row][start_col])
            output.append(adjusted_times)
        if return_warning:
            output.append(warning)
        if verbose and warning is not None:
            print(warning)
        return output[0] if len(output) == 1 else output


class CropMatrix(pd.DataFrame):
    def __init__(self):
        # TODO: is it necessary to have CropParams and other internally-controlled tables in fields matrix?
        super().__init__(self.merge_tables())
        self._double_crops = None

    @staticmethod
    def merge_tables():
        """ Merge all parameter tables linked to crop class """
        from paths import crop_group_path, crop_dates_path, crop_params_path, genclass_params_path, irrigation_path
        return pd.read_csv(crop_group_path) \
            .merge(pd.read_csv(crop_params_path), on='cdl', how='left') \
            .merge(pd.read_csv(crop_dates_path), on='cdl', how='left') \
            .merge(pd.read_csv(irrigation_path), on=['cdl', 'state'], how='left') \
            .merge(pd.read_csv(genclass_params_path), on='gen_class', how='left', suffixes=('_cdl', '_gen'))

    @property
    def double_crops(self):
        if self._double_crops is None:
            self._double_crops = self[~np.isnan(self.double_crop_a)].drop_duplicates(
                ['cdl'])[['cdl', 'double_crop_a', 'double_crop_b']].astype(np.int16)
        return self._double_crops

    def cultivated(self, mode='cdl'):
        return self.matrix['cultivated_' + mode].unique()


class WeatherCube(object):
    def __init__(self, weather_path, years=None, precip_points=None):
        self.storage_path = os.path.join(weather_path, "weather_cube.dat")
        self.key_path = os.path.join(weather_path, "weather_key.npz")
        self.output_header = ["precip", "pet", "temp", "wind"]

        if years is None and precip_points is None:
            self.years, self.precip_points = self.load_key()

    def fetch(self, point_num):
        array = np.memmap(self.storage_path, mode='r', dtype=np.float32, shape=self.shape)
        out_array = array[:, point_num]
        del array
        dates = pd.date_range(self.start_date, self.end_date)
        return pd.DataFrame(data=out_array, columns=self.output_header, index=dates)

    def load_key(self):
        data = np.load(self.key_path)
        return data['years'], pd.DataFrame(data['points'], columns=['lat', 'lon'])

    @property
    def start_date(self):
        return dt.date(self.years[0], 1, 1)

    @property
    def end_date(self):
        return dt.date(self.years[-1], 12, 31)

    @property
    def shape(self):
        return (self.end_date - self.start_date).days + 1, self.precip_points.shape[0], len(self.output_header)


def read_gdb(dbf_file, table_name, input_fields=None):
    """Reads the contents of a dbf table """
    import ogr

    # Initialize file
    driver = ogr.GetDriverByName("OpenFileGDB")
    gdb = driver.Open(dbf_file)

    # parsing layers by index
    tables = {gdb.GetLayerByIndex(i).GetName(): i for i in range(gdb.GetLayerCount())}
    table = gdb.GetLayer(tables[table_name])
    table_def = table.GetLayerDefn()
    table_fields = [table_def.GetFieldDefn(i).GetName() for i in range(table_def.GetFieldCount())]
    if input_fields is None:
        input_fields = table_fields
    else:
        missing_fields = set(input_fields) - set(table_fields)
        if any(missing_fields):
            print("Fields {} not found in table {}".format(", ".join(missing_fields), table_name))
            input_fields = [field for field in input_fields if field not in missing_fields]
    data = np.array([[row.GetField(f) for f in input_fields] for row in table])

    return pd.DataFrame(data=data, columns=input_fields)


def read_dbf(dbf_file, fields='all'):
    from dbfread import DBF, FieldParser

    class MyFieldParser(FieldParser):
        def parse(self, field, data):
            try:
                return FieldParser.parse(self, field, data)
            except ValueError as e:
                print(e)
                return None

    try:
        dbf = DBF(dbf_file)
        table = pd.DataFrame(iter(dbf))
    except ValueError:
        dbf = DBF(dbf_file, parserclass=MyFieldParser)
        table = pd.DataFrame(iter(dbf))

    table.rename(columns={column: column.lower() for column in table.columns}, inplace=True)

    return table


# Initialize field matrix
fields = FieldMatrix()

# Initialize crops matrix
crops = CropMatrix()

# The Universal Soil Loss Equation (USLE) Length/Steepness (LS) Factor lookup matrix (uslels_matrix.csv)
# USLE LS is based on the slope length (columns) and slope % (rows)
# See Table 2 in SAM Scenario Input Parameter documentation. Only lengths up to 150 ft are included in the matrix.
# Source: PRZM 3 manual (Carousel et al, 2005).
uslels_matrix = pd.read_csv(uslels_path, index_col=0).astype(np.float32)
