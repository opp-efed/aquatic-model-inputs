import os
from ftplib import FTP

def find_missing(years, local_path, precip_format, ncep_format, ncep_vars):
    precip_files = set()
    ncep_files = set()
    for year in years:
        precip_files.add(precip_format.format(year))
        for ncep_var in ncep_vars:
            ncep_files.add(ncep_format.format(ncep_var, year))
    existing_files = set(os.listdir(local_path))
    return precip_files - existing_files, ncep_files - existing_files


def get_missing(local_path, remote_path, precip_dir, ncep_dir, missing_precip, missing_ncep):
    def fetch_file(filename):
        print("Downloading file {}...".format(filename))
        try:
            local_filename = os.path.join(local_path, filename)
            print(ftp.pwd(), local_filename)
            with open(local_filename, "wb") as f:
                ftp.retrbinary("RETR " + filename, f.write)
            return True
        except Exception as e:
            print("\tFailed")
            os.remove(local_filename)
            return False

    ftp = FTP(remote_path)  # connect to host, default port
    ftp.login()  # user anonymous, passwd anonymous@
    ftp.cwd('Datasets')

    # Download precip files
    ftp.cwd(precip_dir)
    for precip_file in missing_precip:
        success = fetch_file(precip_file)
        if not success:
            ftp.cwd("RT")
            fetch_file(precip_file)
            ftp.cwd("..")
    ftp.cwd("..")

    # Download NCEP files
    ftp.cwd(ncep_dir)
    for ncep_file in missing_ncep:
        success = fetch_file(ncep_file)
        if not success:
            ftp.cwd("../other_gauss")
            fetch_file(ncep_file)
            ftp.cwd("../surface_gauss")


def main():
    # Set paths
    from paths import met_data_path
    remote_path = "ftp.cdc.noaa.gov"
    precip_dir = "cpc_us_precip"  # /RT
    ncep_dir = "ncep.reanalysis/surface_gauss"
    precip_format = "precip.V1.0.{}.nc"  # year
    ncep_format = "{}.gauss.{}.nc"  # var, year
    ncep_vars = ["tmin.2m", "tmax.2m", "air.2m", "dswrf.sfc", "uwnd.10m", "vwnd.10m", "dswrf.ntat"]
    years = range(1961, 2018)

    # Identify which files need to be acquired
    missing_precip, missing_ncep = \
        find_missing(years, met_data_path, precip_format, ncep_format, ncep_vars)

    # Fetch the missing files
    get_missing(met_data_path, remote_path, precip_dir, ncep_dir, missing_precip, missing_ncep)

main()