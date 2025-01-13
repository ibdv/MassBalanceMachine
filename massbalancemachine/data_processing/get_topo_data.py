"""
This code is taken, and refactored, and inspired from the work performed by: Kamilla Hauknes Sjursen

This method fetches the topographical features (variables of interest), for each stake measurement available,
via the OGGM library.

@Author: Julian Biesheuvel
Email: j.p.biesheuvel@student.tudelft.nl
Date Created: 21/07/2024
"""

import os
import config

import xarray as xr
import pandas as pd
import numpy as np
from oggm import cfg, workflow, tasks
from pyproj import Transformer
from scipy.signal import convolve2d
from queue import Queue
import pyproj as proj


def get_topographical_features(df: pd.DataFrame, output_fname: str,
                               voi: "list[str]",
                               rgi_ids: pd.Series,
                               custom_working_dir:str) -> pd.DataFrame:
    """
    Retrieves topographical features for each stake location using the OGGM library and updates the given
    DataFrame with these features.

    Args:
        df (pd.DataFrame): A DataFrame containing columns with RGI IDs, latitude, and longitude for each stake location.
        output_fname (str): The path to the output CSV file where the updated DataFrame will be saved.
        voi (list of str): A list of variables of interest (e.g., ['slope', 'aspect']) to retrieve from the gridded data.
        rgi_ids (pd.Series): A Series of RGI IDs corresponding to the stake locations in the DataFrame.
        custom_working_dir (str): The path to the custom working directory for OGGM data.
    Returns:
        pd.DataFrame: The updated DataFrame with topographical features added.

    Raises:
        ValueError: If no stakes are found for the region of interest, or if the resulting DataFrame is empty.
    """

    data = df.copy()

    # Get a list of unique RGI IDs
    rgi_ids_list = _get_unique_rgi_ids(rgi_ids)

    # Initialize the OGGM Config
    _initialize_oggm_config(custom_working_dir)

    # Initialize the OGGM Glacier Directory, given the available RGI IDs
    glacier_directories = _initialize_glacier_directories(rgi_ids_list)

    # Get all the latitude and longitude positions for all the stakes (with a
    # valid RGI ID)
    filtered_df = _filter_dataframe(df, rgi_ids_list)
    # Group stakes by RGI ID
    grouped_stakes = _group_stakes_by_rgi_id(filtered_df)

    # RGI ID: RGI123
    #    RGIId  POINT_LAT  POINT_LON
    # 0  RGI123       10.0       20.0
    # 1  RGI123       10.5       20.5

    # Load the gridded data for each glacier available in the OGGM Glacier
    # Directory
    gdirs_gridded = _load_gridded_data(glacier_directories, grouped_stakes)

    # Based on the stake location, find the nearest point on the glacier with
    # recorded topographical features
    _retrieve_topo_features(data, glacier_directories, gdirs_gridded,
                            grouped_stakes, voi)

    # Check if the dataframe is not empty (i.e. no points were found)
    if data.empty:
        raise ValueError(
            "DataFrame is empty, no stakes were found for the region of interest. Please check if your \n"
            "RGIIDs are correct, and your coordinates are in the correct CRS.")

    data.to_csv(output_fname, index=False)

    return data


def get_glacier_mask(df:pd.DataFrame, custom_working_dir:str):
    """Gets glacier xarray from OGGM and masks it over the glacier outline."""
    
    # Initialize the OGGM Config
    _initialize_oggm_config(custom_working_dir)
    
    # Initialize the OGGM Glacier Directory, given the available RGI IDs
    rgi_id = df.RGIId.unique()
    gdirs = _initialize_glacier_directories(rgi_id)
    
    # Get oggm data for that RGI ID
    for gdir in gdirs:
        if gdir.rgi_id == rgi_id[0]:
            break
    with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
        ds = ds.load()
        
    # Create glacier mask
    ds = ds.assign(masked_slope=ds['glacier_mask'] * ds['slope'])
    ds = ds.assign(masked_elev=ds['glacier_mask'] * ds['topo'])
    ds = ds.assign(masked_aspect=ds['glacier_mask'] * ds['aspect'])
    ds = ds.assign(masked_dis=ds['glacier_mask'] * ds['dis_from_border'])
    glacier_indices = np.where(ds['glacier_mask'].values == 1)
    return ds, glacier_indices, gdir
    
def _get_unique_rgi_ids(rgi_ids: pd.Series) -> list:
    """Get the list of unique RGI IDs."""
    return rgi_ids.dropna().unique().tolist()


def _initialize_oggm_config(custom_working_dir):
    """Initialize OGGM configuration."""
    cfg.initialize(logging_level="WARNING")
    cfg.PARAMS["border"] = 10
    cfg.PARAMS["use_multiprocessing"] = True
    cfg.PARAMS["continue_on_error"] = True
    if len(custom_working_dir) == 0:
        current_path = os.getcwd()
        cfg.PATHS["working_dir"] = os.path.join(current_path, "OGGM")
    else:
        cfg.PATHS["working_dir"] = custom_working_dir


def _initialize_glacier_directories(rgi_ids_list: list) -> list:
    """Initialize glacier directories."""
    base_url = config.BASE_URL
    glacier_directories = workflow.init_glacier_directories(
        rgi_ids_list,
        reset=False,
        from_prepro_level=3,
        prepro_base_url=base_url,
        prepro_border=10,
    )

    workflow.execute_entity_task(tasks.gridded_attributes,
                                 glacier_directories,
                                 print_log=True)
    return glacier_directories


def _filter_dataframe(df: pd.DataFrame, rgi_ids_list: list) -> pd.DataFrame:
    """Filter the DataFrame to include only the RGI IDs of interest and select only lat/lon columns."""
    return df.loc[df["RGIId"].isin(rgi_ids_list),
                  ["RGIId", "POINT_LAT", "POINT_LON"]]


def _group_stakes_by_rgi_id(
    filtered_df: pd.DataFrame, ) -> pd.api.typing.DataFrameGroupBy:
    """Group latitude and longitude by RGI ID."""
    return filtered_df.groupby("RGIId", sort=False)


def _load_gridded_data(glacier_directories: list,
                       grouped_stakes: pd.api.typing.DataFrameGroupBy) -> list:
    """Load gridded data for each glacier directory."""
    grouped_rgi_ids = set(grouped_stakes.groups.keys())
    return [
        xr.open_dataset(gdir.get_filepath("gridded_data")).load()
        for gdir in glacier_directories if gdir.rgi_id in grouped_rgi_ids
    ]


def _retrieve_topo_features(
    df: pd.DataFrame,
    glacier_directories: list,
    gdirs_gridded: list,
    grouped_stakes: pd.api.typing.DataFrameGroupBy,
    voi: list,
) -> None:
    """Find the nearest recorded point with topographical features on the glacier for each stake."""

    glac_no = 0
    for gdir, gdir_grid in zip(glacier_directories, gdirs_gridded):
        glac_no += 1
        print("Retrieving data  for glacier: "+ str(glac_no) + " from a total of: " + str(len(gdirs_gridded)))

        lat = grouped_stakes.get_group(gdir.rgi_id)[["POINT_LAT"
                                                     ]].values.flatten()
        lon = grouped_stakes.get_group(gdir.rgi_id)[["POINT_LON"
                                                     ]].values.flatten()

        transformer = Transformer.from_crs(crs_from="EPSG:4326",crs_to=proj.CRS.from_proj4(gdir_grid.attrs['pyproj_srs']),always_xy=True)
        [x,y]=transformer.transform(lon,lat)

        x = xr.DataArray(x, dims=['location'])
        y = xr.DataArray(y, dims=['location'])
        
        topo_data = (gdir_grid.sel(
            x=x, y=y,
            method="nearest")[voi].to_dataframe().reset_index(drop=True))
        
        accums, curvatures, vicinity = _compute_terrain_properties(x,y,gdir_grid)

        df.loc[df["RGIId"] == gdir.rgi_id, voi] = topo_data[voi].values
        df.loc[df["RGIId"] == gdir.rgi_id, "accumulation"] = accums
        df.loc[df["RGIId"] == gdir.rgi_id, "curvature"] = curvatures
        df.loc[df["RGIId"] == gdir.rgi_id, "vicinity"] = vicinity

def _compute_terrain_properties(
        x: xr.DataArray,
        y: xr.DataArray,
        gdir_grid: xr.Dataset):

    """ Extract the curvature and flow accumulation based on neighbouring topography of stake """

    ## Instantiating this once helps with performance.
    #silence = Silence()

    kn = 21 # TODO; what is an appropriate number?
    kernel = np.ones((kn,kn)) 

    stakemask = gdir_grid.glacier_mask.copy()

    accums, curvatures, vicinity = [],[],[]
    for xstake, ystake in zip(x, y):
        # Find grid cell that is closest to the stake
        stakemask.values = np.zeros(np.shape(stakemask.values))	# reset values
        xs = stakemask.x.sel(x=xstake,method="nearest")			
        ys = stakemask.y.sel(y=ystake,method="nearest")

        # Extract local DEM with dimensions kn*kn
        stakemask.loc[dict(x=xs, y=ys)] = 1						
        topopad = np.pad(gdir_grid.topo.values, pad_width=np.int64((kn-1)/2), mode='constant', constant_values=np.nan)
        stakemaskpad = np.pad(stakemask, pad_width=np.int64((kn-1)/2), mode='constant', constant_values=0)
        dem = np.reshape(topopad[np.where(convolve2d(stakemaskpad, kernel, mode='same') > 0)],(kn,kn))

        # Compute curvature and flow accumulation
        curvature = compute_curvature(dem, kn,200)
        accum = flow_accumulation(flowProp(dem,kn,1,200), np.ones_like(dem, dtype=np.float64))
        accum = accum/np.max(accum)

        ## SOMETHING WITH DISTANCE TO SLOPES WITH AVALANCHE POTENTIAL
        xgrid, ygrid = np.meshgrid(gdir_grid.x,gdir_grid.y)
        xavaslope = xgrid[((gdir_grid.slope/np.pi*180 > 30) & (gdir_grid.slope/np.pi*180 < 45) & (gdir_grid.topo > gdir_grid.topo.sel(x = xs, y = ys)))]
        yavaslope = ygrid[((gdir_grid.slope/np.pi*180 > 30) & (gdir_grid.slope/np.pi*180 < 45) & (gdir_grid.topo > gdir_grid.topo.sel(x = xs, y = ys)))]

        if xavaslope.size > 0:
            dists = np.sqrt((xavaslope-xs.values)**2 + (yavaslope-ys.values)**2)            
            vic = np.quantile(dists,0.05)
        else:
             vic = np.nan

        accums.append(accum[int(kn/2-0.5),int(kn/2-0.5)])
        curvatures.append(curvature)
        vicinity.append(vic)

    return accums, curvatures, vicinity

def flowProp(Z,kn,xparam,xydist):
	## CONSTANTS
	L1   = 0.5
	L2   = 0.354
	L = [L1,L2,L1,L2,L1,L2,L1,L2]
	d8x = [-1, -1,  0,  1, 1, 1, 0, -1] # offsets of D8 neighbours, from a central cell
	d8y = [0, -1, -1, -1, 0, 1, 1,  1]
	dr = [1,np.sqrt(2),1,np.sqrt(2),1,np.sqrt(2),1,np.sqrt(2)]
	props = np.zeros((kn,kn,9))-1
	Cs = np.zeros((kn,kn))
	grads = []
	for y in range(Z.shape[0]):
		for x in range(Z.shape[1]):

			if (x == 0) or (y == 0) or (x == kn-1) or (y == kn-1): # If edge cell, but why?
				continue

			e = Z[y,x]
			C = 0

			for n in range(8):
				nx = x+d8x[n]
				ny = y+d8y[n]

				if (x < 0) or (y < 0) or (x >= kn) or (y >= kn):
					continue

				ne = Z[ny,nx]

				# Potential distribution amongst neighbouring cells
				if ne < e:
					rise = e-ne
					run  = dr[n]*xydist			
					grad = rise/run
					#if (grad > np.tan(5/180*np.pi)): #& (grad < np.tan(60/180*np.pi)):
					props[y,x,n+1] = (grad * L[n])**xparam # flow to neighbouring cell n
					C = C + props[y,x,n+1] # total flow that is leaving the cell

			# Actual distribution of water amongst lower lying neighbouring cells
			if C > 0:	
				props[y,x,0] = 0  # if water can flow to neighbouring cells, prop of cell is 0
				C = 1/C

				for n in range(8):
					if props[y,x,n+1] > 0:
						props[y,x,n+1] = props[y,x,n+1]*C # turn absolute values into proportions
					else:
						props[y,x,n+1] = 0
	return props

def flow_accumulation(props, accum):
    
    nshift = [-1,-accum.shape[0]-1,-accum.shape[0],-accum.shape[0]+1,1,accum.shape[0]+1,accum.shape[0],accum.shape[0]-1]
    d8x = [-1, -1,  0,  1, 1, 1, 0, -1] # offsets of D8 neighbours, from a central cell
    d8y = [0, -1, -1, -1, 0, 1, 1,  1]

    # Check that `accum` and `props` have the same dimensions
    if (accum.shape[0] != props.shape[0]) or (accum.shape[1] != props.shape[1]):
        raise RuntimeError("Accumulation array must have the same dimensions as the proportions array!")

    # Create dependencies array, initializing all elements to zero
    deps = np.zeros_like(accum, dtype=np.int8)

    # Loop through internal cells, excluding the borders
    for y in range(1, props.shape[0] - 1):
        for x in range(1, props.shape[1] - 1):
            if np.isnan(props[y,x,0]):  
                continue
            for n in range(0, 8):
                if props[y,x, n+1] > 0:  
                    deps[y+d8y[n],x+d8x[n]] += 1                    

    # Find source cells (cells with zero dependencies)
    q = Queue()
    for i in range(deps.size):
        if deps.flat[i] == 0:
            q.put(i)

    # Calculate flow accumulation
    while not q.empty():
        ci = q.get()
        if np.isnan(props.flat[ci]):  
            continue
        c_accum = accum.flat[ci]  # Access the current cell's accumulation value

        for n in range(0, 8):
            if props[:,:,n+1].flat[ci] <= 0:  # No flow in this direction
                continue

            ni = ci + nshift[n] 
            if np.isnan(props.flat[ni]):  
                continue

            accum.flat[ni] += props[:,:,n+1].flat[ci] * c_accum  # Accumulate flow
            deps.flat[ni] -= 1
            if deps.flat[ni] == 0:
                q.put(ni)
            assert deps.flat[ni] >= 0, "deps should not be negative"  # Ensure no negative dependency count

    # Set no-data values in the accumulation array where props are no-data
    for i in range(props.size):
        if np.isnan(props.flat[i]):
            accum.flat[i] = np.nan 

    return accum

def compute_curvature(Z,kn,xydist):

	n = np.int64(kn/3)
	Z = Z.reshape(Z.shape[0]//n, n, Z.shape[1]//n, n).mean(axis=(1, 3))
	D = ((Z[1][0] + Z[1][2])/2- Z[1][1])/xydist**2
	E = ((Z[0][1] + Z[2][1])/2 - Z[1][1])/xydist**2
	F = (-Z[0][0] + Z[0][2] + Z[2][0]  - Z[2][2])/(4*xydist**2)
	G = (-Z[1][0] + Z[1][2])/(2*xydist)
	H = (Z[0][1] - Z[2][1])/(2*xydist)

	theta = np.arctan(-H/-G)
	dZds = -(G**2+H**2)**0.5

	curv = 2*(D*np.cos(theta)**2+E*np.sin(theta)**2+F*np.cos(theta)*np.sin(theta))		

	return curv