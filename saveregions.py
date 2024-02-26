import os
from collections import defaultdict, namedtuple
from typing import Generic, List, Tuple, TypeVar

import h5py

# import jld2
import numpy as np
import pandas as pd
import toml
from osgeo import gdal
from scipy.io import loadmat
from tqdm import tqdm

T = TypeVar("T")


class RegionType:
    pass


def get_extent(geotransform, shape):
    xres, yres = geotransform[1], geotransform[5]
    xsize, ysize = shape[1], shape[0]
    xmin = geotransform[0]
    ymax = geotransform[3]
    xmax = xmin + (xsize * xres)
    ymin = ymax + (ysize * yres)
    return xmin, ymin, xmax, ymax


def _read_raster(infile, extent_flag="none", dim=1):
    dataset = gdal.Open(infile)
    # print(dataset.GetProjection())
    geotransform = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(dim)
    raster = band.ReadAsArray()
    coord_extent = get_extent(geotransform, raster.shape)

    if extent_flag == "extend_to_full_globe":
        left, bottom, right, top = coord_extent
        xres, yres = geotransform[1], geotransform[5]
        new_width = int(round(360 / xres))
        new_height = int(round(-180 / yres))
        x_indexes = slice(
            int(round((left + 180) / xres)),
            new_width - int(round((180 - right) / xres)),
        )
        y_indexes = slice(
            int(round((top - 90) / yres)), new_height + int(round((bottom + 90) / yres))
        )
        adjusted = np.zeros((new_height, new_width), dtype=raster.dtype)
        adjusted[y_indexes, x_indexes] = raster
        return adjusted, coord_extent
    else:  # extent_flag == 'get_extent'
        return raster, coord_extent


def read_raster(infile, dim=1):
    return _read_raster(infile, "none", dim)[0]


class GADM(Generic[T], RegionType):
    def __init__(self, parentregions: List[T], subregionnames: Tuple[T, ...]):
        self.parentregions = parentregions
        self.subregionnames = subregionnames


def GADM_factory(*regionnames: T) -> GADM:
    return GADM([], *regionnames)


def GADM_parent_child_factory(parentregions: List[T], *subregionnames: T) -> GADM:
    return GADM(parentregions, *subregionnames)


class NUTS(Generic[T], RegionType):
    def __init__(self, subregionnames: Tuple[T, ...]):
        self.subregionnames = subregionnames


def NUTS_factory(*regionnames: T) -> NUTS:
    return NUTS(regionnames)


NOREGION = np.iinfo(np.int16).max


def bbox2indexes(bbox, rasterdensity):
    latindexes = np.round(
        np.flipud(rasterdensity * (90 - bbox[:, 0])) + np.array([[1, 0]])
    ).astype(int)
    lonindexes = np.round(
        rasterdensity * (bbox[:, 1] + 180) + np.array([[1, 0]])
    ).astype(int)
    return latindexes, lonindexes


def longest_circular_sequence(v: List[int], x) -> List[int]:
    longest = []
    current = []
    haswrapped = False
    i = 0
    len_v = len(v)
    sequencelength = lambda seq: 0 if not seq else (seq[1] - seq[0]) % len_v + 1
    while True:
        if v[i] == x:
            if not current:
                current = [i, i]
            else:
                current[1] = i
        else:
            longest = (
                current
                if sequencelength(current) > sequencelength(longest)
                else longest
            )
            if haswrapped:
                return longest
            current = []
        if not haswrapped and i == len_v - 1:
            haswrapped = True
            i = 0
            if sequencelength(current) == len_v:
                return current
        else:
            i += 1
        if i >= len_v:
            break
    return longest


def dataindexes_lat(latdata: List[bool], padding: int = 0):
    len_latdata = len(latdata)
    data_region = np.where(latdata)[0]
    first, last = data_region[0], data_region[-1]
    return range(max(1, first - padding), min(len_latdata, last + padding + 1))


def dataindexes_lon(londata: List[bool], padding: int = 0):
    len_londata = len(londata)
    seq = longest_circular_sequence(londata, False)
    first, last = seq[1] + 1, seq[0] - 1
    last = last if last >= first else last + len_londata
    return [
        ((i - 1) % len_londata) + 1 for i in range(first - padding, last + padding + 1)
    ]


def getconfig(key):
    return _getconfig()[key]


def _getconfig():
    configfile = os.path.join(os.path.expanduser("~"), ".GlobalEnergyGIS_config")
    if not os.path.isfile(configfile):
        raise FileNotFoundError(
            "Configuration file missing, please run saveconfig(datafolder, uid, api_key) first. See GlobalEnergyGIS README."
        )
    return toml.load(configfile)


def in_datafolder(*names):
    return os.path.join(getconfig("datafolder"), *names)


def splitregiondefinitions(regiondefinitionarray):
    regionnames = regiondefinitionarray[:, 0]
    regiondefinitions = [
        (regdef,) if not isinstance(regdef, tuple) else regdef
        for regdef in regiondefinitionarray[:, 1]
    ]
    nutsdef = [
        tuple(rd for rd in regdef if isinstance(rd, NUTS))
        for regdef in regiondefinitions
    ]
    gadmdef = [
        tuple(rd for rd in regdef if isinstance(rd, GADM))
        for regdef in regiondefinitions
    ]
    return regionnames, nutsdef, gadmdef


def makeregions_nuts(region, nuts, subregionnames, regiondefinitions):
    print("Making region index matrix...")
    regionlookup = {
        r: i
        for i, tuptup in enumerate(regiondefinitions, start=1)
        for ntup in tuptup
        for r in ntup["subregionnames"]
    }
    rows, cols = region.shape
    updateprogress = tqdm(total=cols)
    for c in np.random.permutation(cols):
        for r in range(rows):
            nuts_id = nuts[r, c]
            if nuts_id == 0 or region[r, c] > 0:
                continue
            reg = subregionnames[nuts_id]
            while len(reg) >= 2:
                regid = regionlookup.get(reg, 0)
                if regid > 0:
                    region[r, c] = regid
                    break
                reg = reg[:-1]
        updateprogress.update(1)


def read_gadm():
    print("Reading GADM rasters...")
    gadmfields = pd.read_csv(in_datafolder("gadmfields.csv"), header=0)
    imax = gadmfields.iloc[:, 0].max()
    subregionnames = np.full((imax, 3), "", dtype=object)
    subregionnames[gadmfields.iloc[:, 0] - 1] = (
        gadmfields.iloc[:, 1:4].astype(str).values
    )
    gadm = read_raster(in_datafolder("gadm.tif"))
    return gadm, subregionnames


def read_nuts():
    print("Reading NUTS rasters...")
    nutsfields = pd.read_csv(in_datafolder("nutsfields.csv"), header=0)
    imax = nutsfields.iloc[:, 0].max()
    subregionnames = nutsfields.iloc[
        :, 2
    ].values  # indexes of NUTS regions are in order 1:imax, let's use that
    nuts = read_raster(in_datafolder("nuts.tif"))
    return nuts, subregionnames


def roundbbox(bbox, rasterdensity):
    newbbox = np.zeros((2, 2))
    newbbox[0, :] = np.floor(bbox[0, :] * rasterdensity) / rasterdensity
    newbbox[1, :] = np.ceil(bbox[1, :] * rasterdensity) / rasterdensity
    return newbbox


def bbox2ranges(bbox, rasterdensity):
    latindexes, lonindexes = bbox2indexes(bbox, rasterdensity)
    latindex = range(latindexes[0], latindexes[1] + 1)
    lonindex = range(lonindexes[0], lonindexes[1] + 1)
    return latindex, lonindex


def loadregions(regionname):
    with h5py.File(in_datafolder(f"regions_{regionname}.jld"), "r") as file:
        return (
            file["regions"][:],
            file["offshoreregions"][:],
            file["regionlist"][:],
            file["lonrange"][:],
            file["latrange"][:],
        )


def getbboxranges(regions, padding=0):
    data = (regions > 0) & (regions != NOREGION)
    lonrange = dataindexes_lon(
        np.any(data, axis=1), padding
    )  # all longitudes with region data
    latrange = dataindexes_lat(
        np.any(data, axis=0), padding
    )  # all latitudes with region data
    return lonrange, latrange


def makeregions(regiondefinitionarray, allowmixed=False):
    regionnames, nutsdef, gadmdef = splitregiondefinitions(regiondefinitionarray)
    use_nuts = not all(len(nd) == 0 for nd in nutsdef)
    use_gadm = not all(len(gd) == 0 for gd in gadmdef)
    if use_gadm and not use_nuts:
        regiontype = "GADM"
    elif use_nuts and not use_gadm:
        regiontype = "NUTS"
    elif use_nuts and use_gadm:
        regiontype = "MIXED"
    else:
        regiontype = "WEIRD"
    if not allowmixed and regiontype == "MIXED":
        raise ValueError("Sorry, mixed NUTS & GADM definitions are not supported yet.")
    region = np.zeros((36000, 18000), dtype=np.int16)
    if use_nuts:
        nuts, subregionnames = read_nuts()
        makeregions_nuts(region, nuts, subregionnames, nutsdef)
    if use_gadm:
        gadm, subregionnames = read_gadm()
        makeregions_gadm(region, gadm, subregionnames, gadmdef)
    return region, regiontype


def makeregions_gadm(region, gadm, subregionnames, regiondefinitions):
    print("Making region index matrix...")
    regionlookup = build_inverseregionlookup(regiondefinitions)
    rows, cols = region.shape
    for c in tqdm(np.random.permutation(cols)):
        for r in range(rows):
            gadm_uid = gadm[r, c]
            if gadm_uid == 0 or gadm_uid == 78413 or region[r, c] > 0:
                continue
            reg0, reg1, reg2 = subregionnames[gadm_uid]
            regid = lookup_regionnames(regionlookup, reg0, reg1, reg2)
            if regid > 0:
                region[r, c] = regid


def lookup_regionnames(regionlookup, reg0, reg1, reg2):
    v = regionlookup.get((reg0, "*", "*"), 0)
    if v > 0:
        return v
    v = regionlookup.get((reg0, reg1, "*"), 0)
    if v > 0:
        return v
    return regionlookup.get((reg0, reg1, reg2), 0)


def build_inverseregionlookup(regiondefinitions):
    d = defaultdict(int)
    for reg, regdefs in enumerate(regiondefinitions, 1):
        for regdef in regdefs:
            parentregions, subregionnames = regdef.parentregions, regdef.subregionnames
            regions = ["*", "*", "*"]
            regions[: len(parentregions)] = parentregions
            for s in subregionnames:
                regions[len(parentregions)] = s
                d[tuple(regions)] = reg
    return d


def saveregions(
    regionname,
    regiondefinitionarray,
    autocrop=True,
    bbox=np.array([[-90, -180], [90, 180]]),
):
    land = loadmat(in_datafolder("landcover.mat"))["landcover"]
    if not np.all(bbox == np.array([[-90, -180], [90, 180]])):
        autocrop = False  # ignore supplied autocrop option if user changed bbox
    _saveregions(regionname, regiondefinitionarray, land, autocrop, bbox)


def _saveregions(regionname, regiondefinitionarray, landcover, autocrop, bbox):
    regions, regiontype = makeregions(
        regiondefinitionarray, allowmixed=(regionname == "Europe_background")
    )
    if autocrop:
        # get indexes of the bounding box containing onshore region data with 6% of padding
        lonrange, latrange = getbboxranges(regions)
        padding = int(np.amax(regions[lonrange, latrange].shape) * 0.06)
        lonrange, latrange = getbboxranges(regions, padding)
    else:
        latrange, lonrange = bbox2ranges(roundbbox(bbox, 100), 100)
    landcover = landcover[lonrange, latrange]
    regions = regions[lonrange, latrange]
    if regionname != "Global_GADM0" and regionname != "Europe_background":
        if regiontype == "NUTS":
            print(
                "\nNUTS region definitions detected (using Europe_background region file)..."
            )
            europeregions = loadregions("Europe_background")[0][lonrange, latrange]
            regions[(regions == 0) & (europeregions > 0)] = NOREGION
        elif regiontype == "GADM":
            print(
                "\nGADM region definitions detected (using Global_GADM0 region file)..."
            )
            globalregions = loadregions("Global_GADM0")[0][lonrange, latrange]
            regions[(regions == 0) & (globalregions > 0)] = NOREGION
    print("\nAllocate non-region pixels to the nearest region (for offshore wind)...")
    territory = regions[feature_transform(regions > 0)]
    offshoreregions = territory * (landcover == 0)
    if regionname != "Global_GADM0" and regionname != "Europe_background":
        regions = territory * (landcover > 0)
    print("\nSaving regions and offshoreregions...")
    regionlist = [str(x) for x in regiondefinitionarray[:, 0]]
    # TODO
    # jld2.save(
    # in_datafolder(f"regions_{regionname}.jld"),
    # "regions",
    # regions,
    # "offshoreregions",
    # offshoreregions,
    # "regionlist",
    # regionlist,
    # "lonrange",
    # lonrange,
    # "latrange",
    # latrange,
    # compress=True,
    # )
