import os

# Root directories
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "bin"))
external_data_root = os.path.join(r"C:\\", "Users", "jhook", "Documents", "NationalData")
table_dir = os.path.join(root_dir, "Tables")
input_dir = os.path.join(root_dir, "Input")
intermediate_dir = os.path.join(root_dir, "Intermediate")
production_dir = os.path.join(root_dir, "Production")

# Raw input data
nhd_path = os.path.join(external_data_root, "NHDPlusV2", "NHDPlus{}", "NHDPlus{}")  # vpu, region
#soil_path = os.path.join(external_data_root, "CustomSSURGO")
soil_path = r"P:\GIS_Data\ssurgo\gSSURGO_2016\gSSURGO_{0}.gdb\MapunitRaster_{0}_10m"  # state
cdl_path = os.path.join(external_data_root, "CDL", "{0}_30m_cdls", "{0}_30m_cdls.img")  # year
volume_path = os.path.join(input_dir, "LakeMorphometry", "region_{}.dbf")  # region
met_data_path = os.path.join(input_dir, "MetData")
met_poly_path = os.path.join(input_dir, "MetPolygons")
met_poly_layers = ["anetd_poly", "rainfall_poly", "counties_fips"]

# Intermediate datasets
combo_path = os.path.join(intermediate_dir, "Combinations", "{}_{}.npz")  # region, year
met_grid_path = os.path.join(intermediate_dir, "Weather", "met_stations.csv")
condensed_soil_path = os.path.join(intermediate_dir, "CustomSSURGO")
condensed_nhd_path = os.path.join(intermediate_dir, "CustomNHD", "region_{}.npz")  # region
processed_soil_path = os.path.join(intermediate_dir, "ProcessedSoils", "{}", "region_{}.csv")  # mode, region
projected_layers_path = os.path.join(intermediate_dir, "Tifs")
projected_met_path = os.path.join(projected_layers_path, "WeatherGrid.tif")
projected_cdl_path = os.path.join(projected_layers_path, "CDL", "cdl{}.tif")  # year
projected_nhd_path = os.path.join(projected_layers_path, "NHD_Catchments", "region{}.tif")  # region
projected_soil_path = os.path.join(intermediate_dir, "ProjectedLayers", "SSURGO", "{}")  # state

# Table paths
crop_params_path = os.path.join(table_dir, "cdl_params.csv")
genclass_params_path = os.path.join(table_dir, "genclass_params.csv")
crop_dates_path = os.path.join(table_dir, "dummy_dates_092818.csv")
crop_group_path = os.path.join(table_dir, "crop_groups_082418.csv")
met_attributes_path = os.path.join(table_dir, "met_params.csv")
fields_and_qc_path = os.path.join(table_dir, "fields_and_qc.csv")
uslels_path = os.path.join(table_dir, "uslels_matrix.csv")
irrigation_path = os.path.join(table_dir, "irrigation.csv")

# Production data
metfile_path = os.path.join(production_dir, "WeatherFiles")
hydro_file_path = os.path.join(production_dir, "HydroFiles", "region_{}_{}.npz")  # region, type
recipe_path = os.path.join(production_dir, "RecipeFiles", "r{}_{}.npz")  # region, year
scenario_matrix_path = os.path.join(production_dir, "ScenarioMatrices", "{}", "r{}.csv")  # mode, region
pwc_scenario_path = os.path.join(production_dir, "PwcScenarios", "r{}_{}_{}.csv")  # region, year, crop
pwc_metfile_path = os.path.join(production_dir, "PwcMetfiles", "met_{}.csv")  # grid id