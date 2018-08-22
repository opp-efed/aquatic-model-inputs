import os

# Root directories
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "bin"))
external_data_root = os.path.join(r"C:\\", "Users", "jhook", "Documents", "NationalData")
table_dir = os.path.join(root_dir, "Tables")
input_dir = os.path.join(root_dir, "Inputs")
intermediate_dir = os.path.join(root_dir, "Intermediate")
production_dir = os.path.join(root_dir, "Production")

# Raw input data
nhd_path = os.path.join(external_data_root, "NHDPlusV2", "NHDPlus{}", "NHDPlus{}")  # vpu, region
soil_path = os.path.join(external_data_root, "CustomSSURGO")
metfile_path = os.path.join(external_data_root, "Met1991-2015")
cdl_path = os.path.join(external_data_root, "CDL", "{0}_30m_cdls", "{0}_30m_cdls.img")  # year
met_grid_path = os.path.join(external_data_root, "weather_30m_US", "weather_30m")
volume_path = os.path.join(input_dir, "LakeMorphometry", "region_{}.dbf")  # region
met_data_path = os.path.join(input_dir, "MetData")

# Intermediate datasets
combo_path = os.path.join(intermediate_dir, "Combinations", "{}_{}")  # region, year
condensed_soil_path = os.path.join(intermediate_dir, "CustomSSURGO")
condensed_nhd_path = os.path.join(intermediate_dir, "CustomNHD", "region_{}.npz")  # region
processed_soil_path = os.path.join(intermediate_dir, "ProcessedSoils", "{}", "region_{}")  # mode, region
projected_met_path = os.path.join(intermediate_dir, "ProjectedLayers", "weather_grid")
projected_cdl_path = os.path.join(intermediate_dir, "ProjectedLayers", "CDL", "cdl{}")  # year
projected_nhd_path = os.path.join(intermediate_dir, "ProjectedLayers", "NHD_Catchments", "region{}")  # region
projected_soil_path = os.path.join(intermediate_dir, "ProjectedLayers", "SSURGO", "region_{}")  # region

# Table paths
crop_params_path = os.path.join(table_dir, "cdl_params.csv")
curve_numbers_path = os.path.join(table_dir, "curve_numbers.csv")
crop_dates_path = os.path.join(table_dir, "crop_dates_060618.csv")
met_attributes_path = os.path.join(table_dir, "met_data.csv")
endpoint_format_path = os.path.join(table_dir, "endpoint_format.csv")
crop_group_path = os.path.join(table_dir, "CDL_GenClass_Label_Crosswalk062718.csv")
fields_and_qc_path = os.path.join(table_dir, "fields_and_qc.csv")

# Preprocessed data
metfile_path = os.path.join(production_dir, "WeatherFiles")
hydro_file_path = os.path.join(production_dir, "HydroFiles", "region_{}_{}.npz")  # region, type
recipe_path = os.path.join(production_dir, "RecipeFiles", "r{}_{}.npz")  # region, year
scenario_matrix_path = os.path.join(production_dir, "ScenarioMatrices", "{}", "r{}_{}.csv")  # mode, region, year
