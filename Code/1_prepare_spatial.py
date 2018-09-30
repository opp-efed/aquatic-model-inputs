import os
import arcpy
from arcpy.sa import *
import csv

import numpy as np
from parameters import vpus_nhd, nhd_regions, states

arcpy.CheckOutExtension("Spatial")
from collections import defaultdict


def weather_points_to_raster(points_file, out_raster, met_poly_path, met_poly_layers, out_table, overwrite):
    """ Convert a file of weather points into a nearest station raster """
    if overwrite or not arcpy.Exists(out_raster):
        # Read in the weather station points
        arcpy.MakeXYEventLayer_management(points_file, 'lon', 'lat', "points_lyr", arcpy.SpatialReference(4326))
        arcpy.FeatureClassToFeatureClass_conversion("points_lyr", "in_memory", "pointsFC")

        # Intersect the points with polygons
        out_data = defaultdict(dict)
        for layer in met_poly_layers:
            layer_path = os.path.join(met_poly_path, layer + ".shp")
            arcpy.MakeFeatureLayer_management(layer_path, "polygon_lyr")
            arcpy.Intersect_analysis(["in_memory/pointsFC", "polygon_lyr"], "in_memory/isct", output_type="POINT")
            fields = [f.name.lower() for f in arcpy.ListFields("in_memory/isct")
                      if not any([f.name[:3] in ("OID", "FID"), f.name == "Shape"])]
            for row in arcpy.da.SearchCursor("in_memory/isct", fields):
                row = dict(zip(fields, row))
                out_data[row["weather_grid"]].update(row)
            arcpy.Delete_management("polygon_lyr")
            arcpy.Delete_management("in_memory/isct")
        with open(out_table, 'wb') as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for row in out_data.values() for k in row.keys()}))
            writer.writeheader()
            for grid_id, row in out_data.items():
                writer.writerow(row)
        exit()
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
    from paths import nhd_path, cdl_path, met_grid_path, soil_path, met_poly_path, met_poly_layers, \
        met_attributes_path, projected_met_path, projected_cdl_path, projected_nhd_path, projected_soil_path

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
    #weather_points_to_raster(met_grid_path, projected_met_path, met_poly_path, met_poly_layers, met_attributes_path,
    #                         overwrite_met)

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
