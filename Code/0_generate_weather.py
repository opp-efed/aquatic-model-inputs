import os
import xarray as xr
import pandas as pd
import scipy.interpolate
import numpy as np
import datetime as dt

from utilities import WeatherCube


def read_cdf(path):
    return xr.open_dataset(path).to_dataframe()


def map_stations(precip_path, bounds=None, sample_year=1990):
    """ Use a representative precip file to assess the number of precipitation stations """

    # Read file and adjust longitude
    precip_table = read_cdf(precip_path.format(sample_year)).reset_index()
    precip_table['lon'] -= 360

    # Filter out points by geography and completeness
    # This line probably not necessary once we're working with global, right?
    precip_table = precip_table.groupby(['lat', 'lon']).filter(lambda x: x['precip'].sum() > 0)
    if bounds is not None:
        precip_table = \
            precip_table[(precip_table.lat >= bounds[0]) & (precip_table.lat <= bounds[1]) &
                         (precip_table.lon >= bounds[2]) & (precip_table.lon <= bounds[3])]

    # Sort values and add an index
    precip_table = \
        precip_table[['lat', 'lon']].drop_duplicates().sort_values(['lat', 'lon'])

    return precip_table.reset_index(drop=True)


class WeatherCubeGenerator(WeatherCube):
    def __init__(self, scratch_path, years, ncep_vars, ncep_path, precip_path, bounds, precip_points):

        super(WeatherCubeGenerator, self).__init__(scratch_path, years, precip_points)
        self.years = years
        self.precip_points = pd.DataFrame(precip_points, columns=['lat', 'lon'])
        self.populate(ncep_vars, ncep_path, precip_path, bounds)
        self.write_key()

    @staticmethod
    def perform_interpolation(daily_precip, daily_ncep, date):
        daily_precip['date'] = date
        points = daily_ncep[['lat', 'lon']].values
        new_points = daily_precip[['lat', 'lon']].values
        for value_field in ('temp', 'pet', 'wind'):
            daily_precip[value_field] = \
                scipy.interpolate.griddata(points, daily_ncep[value_field].values, new_points)
        return daily_precip

    def populate(self, ncep_vars, ncep_path, precip_path, bounds):

        # Initialize the output in-memory array
        out_array = np.memmap(self.storage_path, mode='w+', dtype=np.float32, shape=self.shape)

        for year in self.years:
            print("Running year {}...\n\tLoading datasets...".format(year))

            # Read, combine, and adjust NCEP tables
            ncep_table = self.read_ncep(year, ncep_path, ncep_vars, bounds)

            # Calculate PET and eliminate unneeded headings
            ncep_table = self.process_ncep(ncep_table)

            # Load precip table
            precip_table = self.read_precip(year, precip_path)

            # Determine the offset in days between the start of the year and the start of all years
            annual_offset = (dt.date(year, 1, 1) - self.start_date).days

            # Loop through each date and perform interpolation
            print("\tPreforming daily interpolation...")
            for i, (date, ncep_group) in enumerate(ncep_table.groupby('date')):
                daily_precip = precip_table[precip_table.time == date]

                # Interpolate NCEP data to resolution of precip data
                daily_table = self.perform_interpolation(daily_precip, ncep_group, date)

                # Write to memory map
                out_array[annual_offset + i] = daily_table[['precip', 'pet', 'temp', 'wind']]

    @staticmethod
    def process_ncep(table):

        def hargreaves_samani(t_min, t_max, solar_rad, temp):
            # ;Convert sradt from W/m2 to mm/d; using 1 MJ/m2-d = 0.408 mm/d per FAO
            # srt1 = (srt(time|:,lat|:,lon|:)/1e6) * 86400. * 0.408
            # ;Hargreaves-Samani Method - PET estimate (mm/day -> cm/day)
            # har = (0.0023*srt1*(tempC+17.8)*(rtemp^0.5))/10

            solar_rad = (solar_rad / 1e6) * 86400. * 0.408
            return ((0.0023 * solar_rad) * (temp + 17.8) * ((t_max - t_min) ** 0.5)) / 10  # (mm/d -> cm/d)

        # Adjust column names
        table.rename(columns={"air": "temp", "dswrf": "solar_rad"}, inplace=True)

        # Convert date-times to dates
        table['date'] = table['time'].dt.normalize()

        # Average out sub-daily data
        table = table.groupby(['lat', 'lon', 'date']).mean().reset_index()

        # Adjust units
        table['temp'] -= 273.15  # K -> deg C

        # Calculate potential evapotranspiration using Hargreaves-Samani method
        table['pet'] = \
            hargreaves_samani(table.pop('tmin'), table.pop('tmax'), table.pop('solar_rad'), table['temp'])

        # Compute vector wind speed from uwind and vwind in m/s to cm/s
        table['wind'] = np.sqrt((table.pop('uwnd') ** 2) + (table.pop('vwnd') ** 2)) * 100.

        return table

    @staticmethod
    def read_ncep(year, ncep_path, ncep_vars, bounds):
        y_min, y_max, x_min, x_max = bounds

        # Read and merge all NCEP data tables for the year
        table_paths = [ncep_path.format(var, year) for var in ncep_vars]
        full_table = None
        for table_path in table_paths:
            table = read_cdf(table_path).reset_index()
            table['lon'] -= 360
            table = table[(table.lat >= y_min) & (table.lat <= y_max) & (table.lon >= x_min) & (table.lon <= x_max)]
            if full_table is None:
                full_table = table
            else:
                full_table = full_table.merge(table, on=['lat', 'lon', 'time'])

        return full_table

    def read_precip(self, year, precip_path):
        precip_table = read_cdf(precip_path.format(year)).reset_index()
        precip_table['lon'] -= 360
        precip_table = self.precip_points.merge(precip_table, how='left', on=['lat', 'lon'])
        return precip_table

    def write_key(self):
        np.savez_compressed(self.key_path, points=self.precip_points, years=np.array(self.years))


def main():
    from paths import met_data_path, met_grid_path, metfile_path

    ncep_vars = ["tmin.2m", "tmax.2m", "air.2m", "dswrf.ntat", "uwnd.10m", "vwnd.10m"]
    ncep_path = os.path.join(met_data_path, "{}.gauss.{}.nc")  # var, year
    precip_path = os.path.join(met_data_path, "precip.V1.0.{}.nc")  # year

    # Specify run parameters
    years = range(1961, 2017)
    bounds = [20, 60, -130, -60]  # min lat, max lat, min long, max long
    precip_points = None
    # Get the coordinates for all precip stations being used and write to file
    #precip_points = map_stations(precip_path, bounds)
    #precip_points.to_csv(met_grid_path, index_label='weather_grid')

    # Process all weather and store to memory
    WeatherCubeGenerator(metfile_path, years, ncep_vars, ncep_path, precip_path, bounds, precip_points)


if __name__ == '__main__':
    main()
