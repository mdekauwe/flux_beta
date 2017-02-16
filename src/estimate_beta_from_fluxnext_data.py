#!/usr/bin/env python3.4
"""
Estimate an effective beta from fluxnet LE data

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (25.01.2017)"
__email__ = "mdekauwe@gmail.com"

#import matplotlib
#matplotlib.use('agg') # stop windows popping up

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import calendar
import datetime as dt
import pandas as pd

from fit_sigmoid_pymc import fitMe, tt_sigmoid, np_sigmoid, make_plot

class FitFluxnetBeta(object):

    def __init__(self, fdir, adir, ofdir, site, rooting_depth=2000):

        self.site = site
        site_fname = "CommonAnc_LATEST.csv"
        self.flux_dir = fdir
        self.flist = glob.glob(os.path.join(self.flux_dir, "*.csv"))
        self.site_fname = os.path.join(adir, site_fname)
        self.rooting_depth = rooting_depth
        # W/m2 = 1000 (kg/m3) * 2.45 (MJ/kg) * 10^6 (J/kg) * 1 mm/day * \
        #        (1/86400) (day/s) * (1/1000) (mm/m)
        # 2.45 * 1E6 W/m2 = kg/m2/s or mm/s
        self.WM2_TO_KG_M2_S = 1.0 / ( 2.45 * 1E6 )
        self.KG_TO_G = 1000.0
        self.MOL_TO_MMOL = 1000.0
        self.G_TO_MOL_H20 = 1.0 / 18.0
        self.HPA_TO_KPA = 0.1
        self.KPA_TO_PA = 1000.0
        self.SEC_2_HFHR = 60.0 * 30.0

    def main(self):

        df_site_info = pd.read_csv(self.site_fname, encoding="ISO-8859-1")

        big_et_store = []
        big_sw_store = []
        for fname in glob.glob(os.path.join(self.flux_dir,
                               "%s.*.synth.hourly.allvars.csv" % (self.site))):



            (site, yr, lat, lon,
             pft, clim_cl, clim_grp,
             name, country) = self.get_site_info(df_site_info, fname)

            #print (site, yr, lat, lon, pft, clim_cl, clim_grp, name, country)
            df = pd.read_csv(fname, index_col='date',
                             parse_dates={'date': ["Year","DoY","Time"]},
                             date_parser=self.date_converter)

            # files contain a rouge date from the following year, fix it.
            df = self.fix_rogue_date(df, drop=True)

            # Convert some units.
            df['VPD_f'] *= self.HPA_TO_KPA # kPa

            # mm / 30 min
            df["ET"] = (df['LE_f'] * self.WM2_TO_KG_M2_S * self.SEC_2_HFHR)

            (ppt, et, sw) = self.estimate_soil_water_bucket(df)

            et_max = et.max()
            et_relative = et / et_max


            i = 0
            N = 10
            while (i < len(ppt)-(N*2)):

                # Find rain event
                if ppt[i] > 0.0:

                    # check if we have at least N days without rain. The N is
                    # arbitary
                    for j in range(i, i+N):
                        if ppt[j] > 0.0:
                            i = j
                            continue
                    else:
                        # ignore the 2 days following rain to minimise soil evap
                        # contributions

                        et_store = []
                        sw_store = []
                        for j in range(i+2, i+N):

                            #print(sw[j], et_relative[j])
                            et_store.append(et_relative[j])
                            sw_store.append(sw[j])
                            big_sw_store.append(sw[j])
                            big_et_store.append(et_relative[j])
                        #plt.plot(sw_store, et_store, "o")

                i = i + 1

            x = np.asarray(big_sw_store)
            y = np.asarray(big_et_store)


        df = pd.DataFrame({'sw':x, 'beta':y})
        #df.to_csv("/Users/mdekauwe/Desktop/crap.csv", index=False)

        # Have points where the soil is full but no or not much ET, screen these
        screen = np.max(df.sw) * 0.9
        #df.beta = np.where(df.sw>screen, np.max(df.beta), df.beta)
        df.beta = np.where(df.sw>screen, 1.0, df.beta)

        fitMe(df, site)



    def estimate_soil_water_bucket(self, df):

        wp = 0.1
        fc = 0.4
        capacity = (fc - wp) * self.rooting_depth

        ppt = df.Precip_f
        ppt = ppt.resample("D").sum()
        ppt[ppt<0.0] = 0.0

        et = df.ET
        et[df['LE_fqcOK'] != 1] = 0.0
        et[et<0.0] = 0.0
        et = et.resample("D").sum()

        sw = pd.Series().reindex_like(et)

        root_zone = capacity
        for i in range(len(ppt)):
            root_zone += ppt[i] - et[i]
            if root_zone < 0.0:
                root_zone = 0.0
            elif root_zone > capacity:
                root_zone = capacity
            sw[i] = root_zone

        return (ppt, et, sw)





    def get_site_info(self, df_site, fname):

        site = os.path.basename(fname).split(".")[0]
        yr = os.path.basename(fname).split(".")[1]
        lat = df_site.loc[df_site.Site_ID == site,'Latitude'].values[0]
        lon = df_site.loc[df_site.Site_ID == site,'Longitude'].values[0]
        pft = df_site.loc[df_site.Site_ID == site,'IGBP_class'].values[0]
        clim_cl = df_site.loc[df_site.Site_ID == site,'Climate_class'].values[0]
        clim_grp = df_site.loc[df_site.Site_ID == site,'Climate_group'].values[0]
        name = df_site.loc[df_site.Site_ID == site,'Name'].values[0]
        country = df_site.loc[df_site.Site_ID == site,'Country'].values[0]

        # remove commas from country tag as it messes out csv output
        name = name.replace("," ,"")

        return (site, yr, lat, lon, pft, clim_cl, clim_grp, name, country)

    def date_converter(self, *args):
        year = int(float(args[0]))
        doy = int(float(args[1]))
        # in leap years the rogue date from the following year will be 367
        # as we are correctin this below it really doesn't matter what we set it to
        # but lets make it 366 for now so that the code doesn't barf
        if doy == 367:
            doy = 366

        hour = int(args[2].split(".")[0])
        minutes = int((float(args[2]) - hour) * 60.)
        date = "%s %s %s:%s" % (year, doy, hour, minutes)

        return pd.datetime.strptime(date, '%Y %j %H:%M')

    def fix_rogue_date(self, df, drop=False):
        files_year = np.median(df.index.year)

        # drop 30 min slot from following year
        if drop:
            df = df[df.index.year == files_year]
        else:
        # files contain a rouge date from the following year, fix it.

            dates = pd.Series([pd.to_datetime(date) for date in df.index]).values
            fixed_date = "%s %s %s:%s" % (int(files_year + 1), 1, 0, 0)
            dates[-1] = pd.datetime.strptime(fixed_date, '%Y %j %H:%M')
            df = df.reindex(dates)

        return df


if __name__ == "__main__":

    #site = "US-Ha1"
    #site = "AU-Tum"
    #site = "DK-Sor"
    site = "FI-Hyy"
    rooting_depth = 2000.
    F = FitFluxnetBeta(fdir="data/raw_data/LaThuile_fluxnet_data/raw_data",
                       adir="data/raw_data/LaThuile_fluxnet_data/ancillary_files/csv/raw/",
                       ofdir="data/processed/",
                       site=site,
                       rooting_depth=rooting_depth)
    F.main()
