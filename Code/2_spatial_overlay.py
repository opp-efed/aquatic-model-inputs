import os
import numpy as np
import pandas as pd
import gdal
import math
import pickle


class CombinationBuilder(object):
    def __init__(self, region, year, weather_raster, cdl_raster, nhd_path, soil_path, out_file, alias_path=None):
        from parameters import vpus_nhd

        # Initialize local variables
        self.region = region
        self.year = year
        self.weather_raster = weather_raster
        self.cdl_raster = cdl_raster
        self.soil_path = soil_path
        self.out_path = out_file

        # Initialize regional NHD raster
        self.nhd_alias = alias_path.format(region)
        self.nhd_raster = Raster(nhd_path.format(vpus_nhd[region], region))
        self.nhd_raster.build_alias(pickle_file=self.nhd_alias)

        # Perform overlay of all layers including state-by-state soils
        output_table = self.overlay_layers()

        self.save_output(output_table)

    def overlay_layers(self):
        from parameters import states_nhd

        out_table = None
        for state in sorted(states_nhd[self.region]):
            print("\tProcessing {}...".format(state))
            state_alias = os.path.join('..', 'Development', 'aliases', 'soils{}.p'.format(state))
            soil_raster = Raster(self.soil_path.format(state), no_data=-2147483647)
            soil_raster.build_alias(pickle_file=state_alias)
            state_table = allocate([self.weather_raster, self.cdl_raster, self.nhd_raster, soil_raster], tile=30000)
            print(sorted(state_table.cdl.unique()))
            state_table.to_csv(state + "_temp.csv")
            out_table = pd.concat([out_table, state_table], axis=0) if out_table is not None else state_table

        index_cols = [col for col in out_table.columns.values if col != 'area']
        out_table = out_table.groupby(index_cols).sum().reset_index()
        return out_table

    def save_output(self, out_table):
        np.savez_compressed(self.out_path, data=out_table.values, columns=out_table.columns.values)


class Envelope(object):
    """ Object representing a simple bounding rectangle, used primarily to measure raster overlap """

    def __init__(self, left, right, bottom, top):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top

    def make_tiles(self, tile_size):
        """ Divides a bounding envelope into smaller tiles for processing on less powerful computers """
        if tile_size == 'max':
            return [self]
        else:
            h = list(range(int(self.left), int(self.right), tile_size)) + [self.right]
            v = list(range(int(self.bottom), int(self.top), tile_size)) + [self.top]
            return [Envelope(h[i], h[i + 1], v[j], v[j + 1]) for i in range(len(h) - 1) for j in range(len(v) - 1)]

    def overlap(self, r2):
        """  Returns the rectangle corresponding to the overlap with another Envelope object """

        def range_overlap(a_min, a_max, b_min, b_max):
            return (a_min <= b_max) and (b_min <= a_max)

        if not all((range_overlap(self.left, self.right, r2.left, r2.right),
                    range_overlap(self.bottom, self.top, r2.bottom, r2.top))):
            return None
        else:
            left, right = sorted([self.left, self.right, r2.left, r2.right])[1:3]
            bottom, top = sorted([self.bottom, self.top, r2.bottom, r2.top])[1:3]
        return Envelope(left, right, bottom, top)

    @property
    def area(self):
        return abs(self.top - self.bottom) * abs(self.right - self.left)

    def __eq__(self, other):
        if (self.left, self.right, self.bottom, self.top) == (other.left, other.right, other.bottom, other.top):
            return True
        else:
            return False

    def __repr__(self):
        return "Rectangle(left: {}, right: {}, top: {}, bottom: {}".format(self.left, self.right, self.top, self.bottom)


class Raster(object):
    """ Wrapper for an ESRI raster grid and function for reading into an array """

    def __init__(self, path, no_data=None, alias=None, no_data_to=0):
        self.path = path
        self.no_data = no_data
        self.obj = gdal.Open(path)
        self.gt = self.obj.GetGeoTransform()
        self.cell_size = int(self.gt[1])
        self.tl = (self.gt[0], self.gt[3])  # top left
        self.size = (self.obj.RasterXSize * self.cell_size, self.obj.RasterYSize * self.cell_size)
        self.shape = Envelope(self.tl[0], self.tl[0] + self.size[0], self.tl[1] - self.size[1], self.tl[1])
        self.obj.GetRasterBand(1).SetNoDataValue(0.0)

        self.envelope = None
        self._array = None
        self.alias = alias

    def array(self, envelope, zero_min=True, dtype=np.int64):
        if self._array is None or (envelope != self.envelope):
            offset_x = (envelope.left - self.shape.left)
            offset_y = (self.shape.top - envelope.top)
            x_max = (envelope.right - envelope.left)
            y_max = (envelope.top - envelope.bottom)
            bounds = map(lambda x: int(x / self.cell_size), (offset_x, offset_y, x_max, y_max))
            self._array = self.obj.ReadAsArray(*bounds)
            if self.alias:
                self._array = np.array([self.alias.get(val, 0) for val in self._array.flat]).reshape(self._array.shape)
            if zero_min:
                self._array[self._array < 0] = 0
            if self.no_data:
                self._array[self._array == self.no_data] = 0
            self.envelope = envelope
        return dtype(self._array)

    def build_alias(self, tile_size=50000, pickle_file=None, overwrite=False):
        if overwrite or not os.path.exists(pickle_file):
            print('\tBuilding an alias for {}...'.format(self.path))
            if not os.path.isdir(os.path.dirname(pickle_file)):
                os.makedirs(os.path.dirname(pickle_file))
            all_values = set()
            tiles = self.shape.make_tiles(tile_size)
            for i, tile in enumerate(tiles):
                all_values |= set(np.unique(self.array(tile)))
            self.alias = dict(zip(sorted(all_values), np.arange(len(all_values))))
            with open(pickle_file, 'wb') as f:
                pickle.dump(self.alias, f)
        else:
            with open(pickle_file, 'rb') as f:
                self.alias = pickle.load(f)

    @property
    def max_val(self):
        if self.alias is None:
            return int(self.obj.GetRasterBand(1).GetMaximum())
        else:
            return max(self.alias.values)

    @property
    def precision(self):
        return 10 ** self.magnitude

    @property
    def magnitude(self):
        return int(math.ceil(math.log10(self.max_val)))


def allocate(all_rasters, tile='max', cell_size=30):
    """ Allocates raster classes to a set of overlapping zones """

    def overlay_tile(rasters):
        combined_array = None
        for r in rasters:

            # Pull the array for the tile
            local_zone = r.array(tile)
            if not local_zone.any():  # If there is no data for the raster in this tile, pull the plug
                return

            # If the raster has a larger cell size than the specified cell size, adjust accordingly
            cellsize_adjust = r.cell_size / cell_size
            if cellsize_adjust > 1:
                local_zone = local_zone.repeat(cellsize_adjust, axis=0).repeat(cellsize_adjust, axis=1)

            # Add the adjusted zone raster to the combined array.
            if combined_array is None:
                combined_array = (local_zone * r.adjust)
            else:
                try:
                    combined_array += (local_zone * r.adjust)
                except ValueError:
                    a0_min = min((local_zone.shape[0], combined_array.shape[0]))
                    a1_min = min((local_zone.shape[1], combined_array.shape[1]))
                    combined_array[:a0_min, :a1_min] += (local_zone[:a0_min, :a1_min] * r.adjust)
                    print("Shape mismatch: {} vs {}. Might be some missing pixels".format(combined_array.shape,
                                                                                          (a0_min, a1_min)))

        return combined_array

    def process_overlay(array, rasters):
        # Break down combined array to get zones and classes
        values, counts = np.unique(array.flat, return_counts=True)  # Cells in each zone and class
        adjust = [r.magnitude for r in rasters]
        bounds = [0] + list(np.cumsum(adjust))
        pairs = list(zip(bounds[:-1], bounds[1:]))
        zones = \
            np.array([[int(str(v).zfill(bounds[-1])[start:end]) for start, end in pairs] for v in values]).T

        # Filter out combinations with no cells
        breakdown = np.vstack((zones, counts))[:, (np.all(zones, axis=0))]

        # Convert back alias
        for i, raster in enumerate(all_rasters):
            if raster.alias is not None:
                convert_back = np.array(list(zip(*sorted(raster.alias.items(), key=lambda x: x[1])))[0])
                breakdown[i] = convert_back[breakdown[i]]

        return breakdown

    # Overlap rasters and create envelope covering common areas
    overlap_area = None
    for i, raster in enumerate(all_rasters):
        overlap_area = raster.shape if overlap_area is None else raster.shape.overlap(overlap_area)
        assert overlap_area, "Zone and allocation rasters do not overlap"

    # Divide the overlap area into tiles to aid in processing
    tiles = overlap_area.make_tiles(tile)

    # Sort by precision and rearrange index
    old_index = [raster.path for raster in all_rasters]
    all_rasters = sorted(all_rasters, key=lambda x: x.precision, reverse=True)
    new_index = [old_index.index(raster.path) for raster in all_rasters]
    header = np.array(["weather_grid", "cdl", "gridcode", "mukey"])[new_index]
    header = list(header) + ['area']

    # Multiply the zone raster by an adjustment factor based on precision
    for i, raster in enumerate(all_rasters):
        raster.adjust = np.prod([r.precision for r in all_rasters[i + 1:]], dtype=np.int64)

    # Iterate through tiles
    finished = None
    for j, tile in enumerate(tiles):
        if not (j + 1) % 10:
            print("\t\tAllocating ({}/{})".format(j + 1, len(tiles)))

        # Perform overlay and process if successful
        tile_array = overlay_tile(all_rasters)
        if tile_array is not None:
            try:
                final = process_overlay(tile_array, all_rasters)
            except IndexError as e:
                print(e)

            # Append to running
            finished = np.hstack((finished, final)) if finished is not None else final
    finished[1] *= (cell_size * cell_size)
    return pd.DataFrame(data=finished.T, columns=header)


def main():
    from parameters import states_nhd

    from paths import projected_nhd_path, projected_soil_path, projected_met_path, projected_cdl_path, \
        alias_path, combo_path

    years = [2010]  # range(2010, 2016)
    regions = sorted(states_nhd.keys())
    overwrite = True

    # Create output directory
    if not os.path.exists(os.path.dirname(combo_path)):
        os.makedirs(os.path.dirname(combo_path))

    # Initialize weather raster
    weather_raster = Raster(projected_met_path)

    # Iterate through year/region combinations
    for year in years:
        cdl_raster = Raster(projected_cdl_path.format(year))
        for region in regions:
            out_file = combo_path.format(region, year) + ".npz"
            if overwrite or not os.path.exists(out_file):
                print("Processing Region {} for {}...".format(region, year))
                try:
                    CombinationBuilder(region, year, weather_raster, cdl_raster, projected_nhd_path,
                                       projected_soil_path, out_file, alias_path)
                except Exception as e:
                    print(e)


main()
