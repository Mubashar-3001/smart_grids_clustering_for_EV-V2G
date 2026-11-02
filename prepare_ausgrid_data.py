"""
Derive the two real-data input files used by both the Python and MATLAB
pipelines from Ausgrid "Solar home electricity data" (customer 12,
1 Jul 2011 - 30 Jun 2012, half-hourly), as redistributed under CC BY 4.0 at
https://github.com/pierre-haessig/ausgrid-solar-data (customer/12/).

Dataset reference: E. L. Ratnam, S. R. Weller, C. M. Kellett, A. T. Murray,
"Residential load and rooftop PV generation: an Australian distribution
network dataset," Int. J. Sustain. Energy, 36(8), 787-806, 2017.

Outputs (written next to this script):
  ausgrid_c12_pv_pu.csv     one row per real day (366 rows):
                            year, month, day, weekday(0=Mon), pv_00 ... pv_47
                            pv_k = gross PV generation in half-hour k divided by
                            the record's 99.5th percentile, clipped to [0, 1]
  ausgrid_c12_load_shape.csv 12 rows (Jan..Dec) x 48 half-hours: monthly mean
                            gross consumption profile, divided by its overall
                            maximum (peak = 1)

Usage: python prepare_ausgrid_data.py <path/to/customer/12/data_2011-2012.csv>
"""
import os
import sys

import numpy as np
import pandas as pd


def main(src):
    here = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(src, index_col=0, parse_dates=True)
    assert {"GC", "GG"} <= set(df.columns)
    df["date"] = df.index.normalize()
    df["slot"] = df.index.hour * 2 + df.index.minute // 30

    # ---- PV: per-day half-hourly matrix, per-unit of the 99.5th percentile
    gg_ref = df["GG"].quantile(0.995)
    pv = df.pivot_table(index="date", columns="slot", values="GG", aggfunc="first")
    pv = (pv / gg_ref).clip(0.0, 1.0).fillna(0.0)
    dates = pd.to_datetime(pv.index)
    out = np.column_stack([dates.year, dates.month, dates.day, dates.dayofweek,
                           pv.to_numpy()])
    hdr = "year,month,day,weekday," + ",".join(f"pv_{k:02d}" for k in range(48))
    np.savetxt(os.path.join(here, "ausgrid_c12_pv_pu.csv"), out, delimiter=",",
               header=hdr, comments="",
               fmt=["%d", "%d", "%d", "%d"] + ["%.4f"] * 48)

    # ---- Load: monthly mean consumption shape, normalised to peak = 1
    df["month"] = df.index.month
    shape = df.pivot_table(index="month", columns="slot", values="GC", aggfunc="mean")
    shape = shape / shape.to_numpy().max()
    np.savetxt(os.path.join(here, "ausgrid_c12_load_shape.csv"), shape.to_numpy(),
               delimiter=",", header=",".join(f"h_{k:02d}" for k in range(48)),
               comments="", fmt="%.4f")

    print(f"days={len(pv)}  PV ref (99.5 pct)={gg_ref:.3f} kWh/30min  "
          f"load shape min/max={shape.to_numpy().min():.3f}/1.000")


if __name__ == "__main__":
    main(sys.argv[1])
