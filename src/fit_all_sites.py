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
from estimate_beta_from_fluxnext_data import FitFluxnetBeta

class FitAll(FitFluxnetBeta):

    def __init__(self, fdir, adir, ofdir, site, rooting_depth):
        FitFluxnetBeta.__init__(self, fdir, adir, ofdir, site, rooting_depth)

    def main(self):

        df_site_info = pd.read_csv(self.site_fname, encoding="ISO-8859-1")

        # Grab unique file list so that we can loop over all sites
        sites = glob.glob(os.path.join(self.flux_dir, "*.csv"))
        sites = [os.path.basename(s) for s in sites]
        sites = [s[0:6] for s in sites]
        sites = np.sort(np.unique(sites))

        for site in sites:
            print(site)
            big_et_store = []
            big_sw_store = []
            for fname in glob.glob(os.path.join(self.flux_dir,
                                   "%s.*.synth.hourly.allvars.csv" % (site))):

                (site, yr, lat, lon,
                 pft, clim_cl, clim_grp,
                 name, country) = self.get_site_info(df_site_info, fname)

                df = pd.read_csv(fname, index_col='date',
                                 parse_dates={'date': ["Year","DoY","Time"]},
                                 date_parser=self.date_converter)

                # files contain a rouge date from the following year, fix it.
                df = self.fix_rogue_date(df, drop=True)

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

                        # check if we have at least N days without rain.
                        # The N is arbitary
                        for j in range(i, i+N):
                            if ppt[j] > 0.0:
                                i = j
                                continue
                        else:
                            # ignore the 2 days following rain to minimise soil
                            # evap contributions

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

            # Have points where the soil is full but no or not much ET, screen
            # these
            screen = np.max(df.sw) * 0.9
            #df.beta = np.where(df.sw>screen, np.max(df.beta), df.beta)
            df.beta = np.where(df.sw>screen, 1.0, df.beta)

            x_range = np.linspace(0, 1000, 100)
            fitMe(df, site, x_range, to_screen=False)


if __name__ == "__main__":


    site = "blah"
    rooting_depth = 2000.
    F = FitAll(fdir="data/raw_data/LaThuile_fluxnet_data/raw_data",
               adir="data/raw_data/LaThuile_fluxnet_data/ancillary_files/csv/raw/",
               ofdir="data/processed/",
               site=site,
               rooting_depth=rooting_depth)
    F.main()
