import os
import arcpy
from arcpy.sa import *
import numpy as np
from parameters import vpus_nhd, nhd_regions, states

arcpy.CheckOutExtension("Spatial")


def weather_points_to_raster(points_file, out_raster, state_layer, met_state_table, overwrite=True):
    """ Convert a file of weather points into a nearest station raster """
    fields = ["state", "county", "GEOID"]  # can't bring in fields_and_qc because of pandas/arcgis incompat

    if overwrite or not arcpy.Exists(out_raster):
        # Read in the weather station points
        arcpy.MakeXYEventLayer_management(points_file, 'lon', 'lat', "points_lyr", arcpy.SpatialReference(4326))
        arcpy.FeatureClassToFeatureClass_conversion("points_lyr", "in_memory", "pointsFC")

        # Intersect the points with states
        arcpy.MakeFeatureLayer_management(state_layer, "states_lyr")
        arcpy.Intersect_analysis(["in_memory/pointsFC", "states_lyr"], "in_memory/states_isct", output_type="POINT")
        print([f.name for f in arcpy.ListFields("in_memory/states_isct")])
        with open(met_state_table, 'w') as f:
            f.write("grid_id,state,county,fips\n")
            for row in arcpy.da.SearchCursor("in_memory/states_isct", ['grid_id'] + fields):
                f.write("{},{},{},{}\n".format(*row))

        # Create a nearest-point raster
        arcpy.CreateThiessenPolygons_analysis("in_memory/pointsFC", "in_memory/tp", "ALL")
        arcpy.FeatureToRaster_conversion("in_memory/tp", "grid_id", out_raster)
        arcpy.Delete_management("in_memory/tp")
        arcpy.Delete_management("in_mempory/pointsFC")


def project_raster(in_path, out_path, sample_raster, overwrite, resample_soil=False):
    if overwrite or not os.path.exists(out_path):
        if arcpy.Exists(out_path):
            arcpy.Delete_management(out_path)
        else:
            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path))
        raster = arcpy.Raster(in_path)
        if resample_soil:
            arcpy.ProjectRaster_management(raster, out_path, sample_raster, "NEAREST", cell_size=30)
        else:
            arcpy.ProjectRaster_management(raster, out_path, sample_raster, "NEAREST")


def main():
    from paths import nhd_path, cdl_path, met_grid_path, soil_path, boundaries_path, \
        projected_met_path, projected_cdl_path, projected_nhd_path, projected_soil_path, met_to_geo_path

    # Set run parameters
    years = range(2010, 2017)
    overwrite_met = True
    overwrite_cdl = False
    overwrite_nhd = False
    overwrite_ssurgo = False

    # Pick a sample SSURGO raster to use as template and set environments
    sample_raster = arcpy.Raster(soil_path.format('al'))
    arcpy.env.snapRaster = sample_raster
    arcpy.env.outputCoordinateSystem = sample_raster
    arcpy.env.cellSize = 30

    # Create weather grid
    print("Creating weather grid...")
    weather_points_to_raster(met_grid_path, projected_met_path, boundaries_path, met_to_geo_path, overwrite_met)

    """
    # CDL Rasters
    for year in years:
        print("Projecting CDL for {}...".format(year))
        project_raster(cdl_path.format(year), projected_cdl_path.format(year), sample_raster, overwrite_cdl)

    # NHD Catchments
    for region in nhd_regions:
        print("Projecting catchments for region {}...".format(region))
        project_raster(nhd_path.format(vpus_nhd[region], region), projected_nhd_path.format(region), sample_raster,
                       overwrite_nhd)

    # Soils
    for state in states:
        print("Projecting SSURGO for {}...".format(state))
        project_raster(soil_path.format(state), projected_soil_path.format(state), sample_raster, overwrite_ssurgo,
                       resample_soil=True)
    """

main()
