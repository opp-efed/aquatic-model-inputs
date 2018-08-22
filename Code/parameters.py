import os
import numpy as np
import pandas as pd
from collections import OrderedDict

# Hydrologic soil groups
hydro_soil_groups = ['A', 'A/D', 'B', 'B/D', 'C', 'C/D', 'D']

# Soil depth bins;
depth_bins = np.array([5, 20, 50, 100])

# The Universal Soil Loss Equation (USLE) Length/Steepness (LS) Factor lookup matrix (uslels_matrix.csv)
# USLE LS is based on the slope length (columns) and slope % (rows)
# See Table 2 in SAM Scenario Input Parameter documentation. Only lengths up to 150 ft are included in the matrix.
# Source: PRZM 3 manual (Carousel et al, 2005).
uslels_matrix = pd.read_csv(os.path.join("..", "bin", "Tables", "uslels_matrix.csv"), index_col=0).astype(np.float32)

# USLEP (practices) values for aggregation based on Table 4 in SAM Scenario Input Parameter documentation.
# Original source: Table 5.6 in PRZM 3 Manual (Carousel et al, 2015).
# USLEP values for cultivated crops by slope bin (0-2, 2-5, 5-10, 10-15, 15-25, >25)
uslep_values = [0.6, 0.5, 0.5, 0.6, 0.8, 0.9]

# Aggegation bins for soil map units (see Addendum E of SAM Scenario Input Parameter Documentation)
bins = {'slope': [0, 2, 5, 10, 15, 25, 200],
        'orgC_5': [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 12, 20, 100],
        'sand_5': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'clay_5': [0, 5, 10, 15, 20, 25, 30, 40, 60, 80, 100]}

# Double crops identified in CDL (General Crop Groups). See Addendum E of SAM Scenario Input Parameter Documentation.
# First crop group listed is used for runoff/erosion generation. Both crops are available for pesticide app/runoff.
double_crops = {14: [10, 40], 15: [10, 24], 18: [10, 80], 25: [20, 24], 26: [20, 60], 42: [40, 20], 45: [40, 24],
                48: [40, 80], 56: [60, 24], 58: [24, 80], 68: [60, 80]}

cultivated_crops = [10, 14, 15, 18, 20, 22, 23, 24, 25, 26, 40, 42, 45, 48, 56, 58, 60, 61, 68, 70, 80, 90, 100]

# Check on this range: Is this for uslep_values? (Table 4 in the input documentation breaks slopes by these values)
# Original thought on uslep was to use the slope aggregation bins defined above as approximations. Either way will do.
slope_range = np.array([-1, 2.0, 7.0, 12.0, 18.0, 24.0])

erom_months = [str(x).zfill(2) for x in range(1, 13)] + ['ma']

# Values are from Table F1 of TR-55 (tr_55.csv), interpolated values are included to make arrays same size
# type column is rainfall parameter (corresponds to IREG in PRZM5 manual) found in met_data.csv
# rainfall is based on Figure 3.3 from PRZM5 manual (Young and Fry, 2016), digitized and joined with weather grid ID
# Map source: Appendix B in USDA (1986). Urban Hydrology for Small Watersheds, USDA TR-55.
# Used in the model to calculate time of concentration of peak flow for use in erosion estimation.
# met_data.csv comes from Table 4.1 in the PRZM5 Manual (Young and Fry, 2016)
types = pd.read_csv(os.path.join("..", "bin", "Tables", "tr_55.csv"), dtype=np.float32)

# NHD regions and the states that overlap
states_nhd = OrderedDict((('01', {"ME", "NH", "VT", "MA", "CT", "RI", "NY"}),
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

vpus_nhd = {'01': 'NE', '02': 'MA', '03N': 'SA', '03S': 'SA', '03W': 'SA', '04': 'GL', '05': 'MS',
            '06': 'MS', '07': 'MS', '08': 'MS', '09': 'SR', '10L': 'MS', '10U': 'MS', '11': 'MS',
            '12': 'TX', '13': 'RG', '14': 'CO', '15': 'CO', '16': 'GB', '17': 'PN', '18': 'CA'}


# All states
nhd_regions = sorted(states_nhd.keys())
states = sorted(set().union(*states_nhd.values()))
