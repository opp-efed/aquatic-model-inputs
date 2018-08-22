import os
import arcpy
from arcpy.sa import *
from collections import OrderedDict

# Can this be de-ESRIfied?
arcpy.CheckOutExtension("Spatial")


def project_raster(in_path, out_path, sample_raster, overwrite, resample_soil=False):
    if overwrite or not os.path.exists(out_path):
        if arcpy.Exists(out_path):
            arcpy.Delete_management(out_path)
        else:
            print(out_path)
            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path))
        raster = arcpy.Raster(in_path)
        if resample_soil:
            arcpy.ProjectRaster_management(raster, out_path, sample_raster, "NEAREST", cell_size=30)
        else:
            arcpy.ProjectRaster_management(raster, out_path, sample_raster, "NEAREST")


def main():
    # from Preprocessing.utilities import nhd_states, states
    from paths import nhd_path, cdl_path, met_grid_path, soil_path
    from paths import projected_met_path, projected_cdl_path, projected_nhd_path, projected_soil_path

    # Set input paths
    soil_path = os.path.join(soil_path, "{0}", "{0}")

    years = range(2010, 2017)
    overwrite_nhd = False
    overwrite_cdl = False
    overwrite_met = False
    overwrite_ssurgo = False

    # Pick a sample SSURGO raster to use as template
    state_soil_raster = arcpy.Raster(soil_path.format('AL'))
    arcpy.env.snapRaster = state_soil_raster
    arcpy.env.outputCoordinateSystem = state_soil_raster

    # Project weather grid
    print("Projecting weather grid...")
    project_raster(met_grid_path, projected_met_path, state_soil_raster, overwrite_met)

    # Project CDL rasters
    for year in years:
        print("Projecting CDL for {}...".format(year))
        project_raster(cdl_path.format(year), projected_cdl_path.format(year), state_soil_raster, overwrite_cdl)

    for region, states in nhd_states.items():
        print("Projecting catchments for region {}...".format(region))
        project_raster(nhd_path.format(vpus_nhd[region], region), projected_nhd_path.format(region), state_soil_raster,
                       overwrite_nhd)

    for state in all_states:
        print("Projecting SSURGO for {}...".format(state))
        project_raster(soil_path.format(state), projected_soil_path.format(state), state_soil_raster, overwrite_ssurgo,
                       resample_soil=True)


# NHD regions and the states that overlap
nhd_states = OrderedDict((('01', {"ME", "NH", "VT", "MA", "CT", "RI", "NY"}),
                          ('02', {"VT", "NY", "PA", "NJ", "MD", "DE", "WV", "DC", "VA"}),
                          ('03N', {"VA", "NC", "SC", "GA"}),
                          ('03S', {"FL", "GA"}),
                          ('03W', {"FL", "GA", "TN", "AL", "MS"}),
                          ('04', {"WI", "MN", "MI", "IL", "IN", "OH", "PA", "NY"}),
                          ('05', {"IL", "IN", "OH", "PA", "WV", "VA", "KY", "TN"}),
                          ('06', {"VA", "KY", "TN", "NC", "GA", "AL", "MS"}),
                          ('07', {"MN", "WI", "SD", "IA", "IL", "MO", "IN"}),
                          ('08', {"MO", "KY", "TN", "AR", "MS", "LA"}),
                          ('09', {"ND", "MN", "SD"}),
                          ('10U', {"MT", "ND", "WY", "SD", "MN", "NE", "IA"}),
                          ('10L', {"CO", "WY", "MN", "NE", "IA", "KS", "MO"}),
                          ('11', {"CO", "KS", "MO", "NM", "TX", "OK", "AR", "LA"}),
                          ('12', {"NM", "TX", "LA"}),
                          ('13', {"CO", "NM", "TX"}),
                          ('14', {"WY", "UT", "CO", "AZ", "NM"}),
                          ('15', {"NV", "UT", "AZ", "NM", "CA"}),
                          ('16', {"CA", "OR", "ID", "WY", "NV", "UT"}),
                          ('17', {"WA", "ID", "MT", "OR", "WY", "UT", "NV"}),
                          ('18', {"OR", "NV", "CA"})))

vpus_nhd = {'01': 'NE',
            '02': 'MA',
            '03N': 'SA',
            '03S': 'SA',
            '03W': 'SA',
            '04': 'GL',
            '05': 'MS',
            '06': 'MS',
            '07': 'MS',
            '08': 'MS',
            '09': 'SR',
            '10L': 'MS',
            '10U': 'MS',
            '11': 'MS',
            '12': 'TX',
            '13': 'RG',
            '14': 'CO',
            '15': 'CO',
            '16': 'GB',
            '17': 'PN',
            '18': 'CA'}


# All states
all_states = sorted(set().union(*nhd_states.values()))

main()
