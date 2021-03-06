import os
from collections import defaultdict

import arcpy
import numpy as np

from parameters import vpus_nhd


def overlay_rasters(*rasters):
    fields = [os.path.splitext(os.path.basename(r))[0].upper() for r in rasters]
    for raster in map(arcpy.Raster, rasters):
        if not raster.hasRAT:
            print("\tBuilding RAT for {}".format(raster.catalogPath))
            arcpy.BuildRasterAttributeTable_management(raster)
    combined = arcpy.sa.Combine(rasters)
    arcpy.MakeRasterLayer_management(combined, "temp")
    arcpy.BuildRasterAttributeTable_management("temp")
    return np.array([row for row in arcpy.da.SearchCursor("temp", fields + ["COUNT"])])


def condense_array(table, cell_size):
    new_table = defaultdict(int)
    for row in table:
        new_table[tuple(row[:4])] += (int(row[4]) * (cell_size ** 2))
    return [list(key) + [area] for key, area in dict(new_table).items()]


def save_table(table, outfile):
    out_header = np.array(['gridcode', 'weather_grid', 'cdl', 'mukey', 'area'])
    np.savez_compressed(outfile, data=table, columns=out_header)


def main():
    from parameters import states_nhd

    from paths import nhd_raster_path, soil_raster_path, weather_raster_path, cdl_path, combo_path

    years = [2010]  # range(2010, 2016)
    arcpy.CheckOutExtension("Spatial")
    arcpy.env.overwriteOutput = True
    cell_size = 30

    # Create output directory
    if not os.path.exists(os.path.dirname(combo_path)):
        os.makedirs(os.path.dirname(combo_path))

    # Initialize weather raster
    nhd_regions = ['07']

    # Iterate through year/region combinations

    for region in nhd_regions:
        nhd_raster = nhd_raster_path.format(vpus_nhd[region], region)
        for year in years:
            cdl_raster = cdl_path.format(region, year)
            master_array = np.zeros((0, 5))
            for state in states_nhd[region]:
                print("Performing overlay for {}, {}, {}".format(year, region, state))
                ssurgo_raster = soil_raster_path.format(state.lower())
                table = overlay_rasters(nhd_raster, weather_raster_path, cdl_raster, ssurgo_raster)
                master_array = np.concatenate((master_array, table))
            out_array = condense_array(master_array, cell_size)
            save_table(out_array, combo_path.format(region, year))


main()
