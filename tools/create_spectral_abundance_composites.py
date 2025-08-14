import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
sys.path.append('../modules/')
import emit_tools as et
import rasterio
import yaml

print("Starting EMIT Spectral Abundance Composite Generation...")

# === CONFIGURATION ===
print("Loading configuration from config.yaml...")
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

input_folder = config["l2b_edge_masked"]
mineral_grouping_matrix_csv = config["mineral_grouping_matrix_csv"]
output_folder = config["spectral_abundance_composites"]

print(f"Input folder: {input_folder}")
print(f"Output folder: {output_folder}")
print(f"Mineral grouping matrix: {mineral_grouping_matrix_csv}")

# Load and prepare mineral grouping matrix
print("\nLoading mineral grouping matrix...")
mineral_groupings = pd.read_csv(mineral_grouping_matrix_csv)
print(f"Original matrix shape: {mineral_groupings.shape}")

# The EMIT 10 Minerals are in columns 6 - 17. Columns after 17 are experimental, and we'll drop for this tutorial:
mineral_groupings = mineral_groupings.drop([x for _x, x in enumerate(mineral_groupings) if _x >= 17], axis=1)
print(f"Matrix shape after dropping experimental columns: {mineral_groupings.shape}")

# Retrieve the EMIT 10 Mineral Names from Columns 7-16 (starting with 0) in .csv
mineral_names = [x for _x, x in enumerate(list(mineral_groupings)) if _x > 6 and _x < 17]
print(f"EMIT 10 Mineral names: {mineral_names}")

# Use EMIT 10 Mineral Names to Subset .csv to only columns with EMIT 10 mineral_names
mineral_abundance_ref = np.array(mineral_groupings[mineral_names])

# Replace Some values in the .csv (NaN -> 0, -1 -> 1)
mineral_abundance_ref[np.isnan(mineral_abundance_ref)] = 0
mineral_abundance_ref[mineral_abundance_ref == -1] = 1
print(f"Mineral abundance reference matrix shape: {mineral_abundance_ref.shape}")

# List .nc files in the directory
def list_nc_files(directory):
    """Recursively find all .nc files in directory and subdirectories"""
    nc_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nc'):
                nc_files.append(os.path.join(root, file))
    return nc_files

print(f"\nScanning for NetCDF files in {input_folder}...")
nc_files_list = list_nc_files(input_folder)
print(f"Found {len(nc_files_list)} NetCDF files to process")

# Create output directories
print(f"\nCreating output directories...")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
subdirs = [
    'abundance',           # Individual mineral abundance rasters
    'abundance_composite', # Multi-band composite rasters
    'g_1_min_id',         # Group 1 mineral ID rasters
    'g_2_min_id',         # Group 2 mineral ID rasters
    'g_1_band_d',         # Group 1 band depth rasters
    'g_2_band_d'          # Group 2 band depth rasters
]

for sub in subdirs:
    os.makedirs(os.path.join(output_folder, sub), exist_ok=True)
    print(f"  Created: {os.path.join(output_folder, sub)}")

# Process each NetCDF file
for file_idx, fp in enumerate(nc_files_list, 1):
    print(f"\n{'='*60}")
    print(f"Processing file {file_idx}/{len(nc_files_list)}: {os.path.basename(fp)}")
    print(f"{'='*60}")
    
    # Load EMIT L2B data as xarray dataset
    print("Loading EMIT L2B data...")
    ds_min = et.emit_xarray(fp)
    
    # Create mineral dataframe with mineral names and add 'No_Match' entry at index 0
    print("Preparing mineral lookup table...")
    min_df = pd.DataFrame({x: ds_min[x].values for x in [var for var in ds_min.coords if 'mineral_name' in ds_min[var].dims]})
    min_df.loc[-1] = {'index': 0, 'mineral_name': 'No_Match', 'record': -1.0, 'url': 'NA', 'group': 1.0, 'library': 'NA', 'spatial_ref': 0}
    min_df = min_df.sort_index().reset_index(drop=True)
    
    # Ortho-rectify the dataset to geographic coordinates
    print("Ortho-rectifying dataset...")
    ds_min = et.ortho_xr(ds_min)
    print(f"Dataset shape after ortho-rectification: {ds_min['group_1_band_depth'].shape}")
    
    # Convert fill values (-9999) to NaN for proper handling
    print("Converting fill values to NaN...")
    for var in ds_min.data_vars:
        ds_min[var].data[ds_min[var].data == -9999] = np.nan
    
    # Calculate mineral abundance for each pixel
    print("Calculating mineral abundance maps...")
    mineral_abundance = np.zeros((ds_min['group_1_band_depth'].shape[0], ds_min['group_1_band_depth'].shape[1], len(mineral_names)))
    
    for _m, mineral_name in enumerate(mineral_names):
        print(f"  Processing {mineral_name} ({_m+1}/{len(mineral_names)})...")
        
        # For each row in the mineral grouping matrix
        for _c in range(mineral_groupings.shape[0]):
            if np.isnan(mineral_groupings[mineral_name][_c]) == False:      
                group = mineral_groupings["Group"][_c]
                # Accumulate abundance: (pixel matches mineral ID) * band_depth * reference_abundance
                mineral_abundance[...,_m] += (ds_min[f'group_{group}_mineral_id'].values == mineral_groupings['Index'][_c]) * ds_min[f'group_{group}_band_depth'].values * mineral_abundance_ref[_c][_m]
    
    # Clean up abundance array: set isolated zeros to NaN, keep real zeros
    mineral_abundance[np.isnan(mineral_abundance)] = 0
    mineral_abundance[np.all(mineral_abundance == 0, axis=-1),:] = np.nan
    
    # Convert to xarray dataset for consistent handling and metadata preservation
    print("Converting abundance arrays to xarray dataset...")
    mineral_abundance_xarray = xr.merge([xr.DataArray(mineral_abundance[...,_x],
                                                      name=mineral_names[_x],
                                                      coords=ds_min['group_1_band_depth'].coords,
                                                      attrs=ds_min.attrs,) 
                                         for _x in range(len(mineral_names))])

    # Save individual mineral abundance rasters
    print("Saving individual mineral abundance rasters...")
    for mineral in mineral_names:
        out_name = f'{mineral_abundance_xarray.granule_id}_{mineral}.tif'
        print(f"  Saving: {out_name}")
        dat_out = mineral_abundance_xarray[mineral]    
        dat_out.data = np.nan_to_num(dat_out.data, nan=-9999)    
        dat_out.rio.write_nodata(-9999, encoded=True, inplace=True)    
        dat_out.rio.to_raster(raster_path=os.path.join(output_folder, 'abundance', out_name), driver='COG')
    
    # Save multi-band composite raster
    print("Saving multi-band abundance composite...")
    out_name = f'{mineral_abundance_xarray.granule_id}_abundance_composite.tif'
    composite_path = os.path.join(output_folder, 'abundance_composite', out_name)
    print(f"  Saving: {out_name}")
    
    # Concatenate all mineral bands into a single multi-band raster
    cdata_out = xr.concat([mineral_abundance_xarray[name] for name in mineral_names], dim='band')
    cdata_out.coords['band'] = mineral_names
    cdata_out.data = np.nan_to_num(cdata_out.data, nan=-9999)
    cdata_out.rio.write_nodata(-9999, encoded=True, inplace=True)
    cdata_out.rio.to_raster(composite_path, driver='COG')

    # Set band descriptions for the composite (requires breaking COG layout)
    print("  Setting band descriptions...")
    with rasterio.open(composite_path, 'r+', IGNORE_COG_LAYOUT_BREAK='YES') as dst:
        for i, name in enumerate(mineral_names):
            dst.set_band_description(i + 1, name)

    # Optionally save original L2B products as GeoTIFF
    if config.get("convert_L2b_products_to_geotiff") == True:
        print("Converting L2B products to GeoTIFF format...")
        
        # Group 1 Mineral ID
        print("  Saving Group 1 Mineral ID...")
        out_name = f'{ds_min.granule_id}_group_1_mineral_id.tif'    
        dat_out = ds_min['group_1_mineral_id']
        dat_out.data = np.nan_to_num(dat_out.data, nan=-9999)
        dat_out.data = dat_out.data.astype(int)
        dat_out.rio.write_nodata(-9999, encoded=True, inplace=True)
        dat_out.rio.to_raster(raster_path=os.path.join(output_folder, 'g_1_min_id', out_name), driver='COG')
        
        # Group 2 Mineral ID
        print("  Saving Group 2 Mineral ID...")
        out_name = f'{ds_min.granule_id}_group_2_mineral_id.tif'    
        dat_out = ds_min['group_2_mineral_id']
        dat_out.data = np.nan_to_num(dat_out.data, nan=-9999)
        dat_out.data = dat_out.data.astype(int)
        dat_out.rio.write_nodata(-9999, encoded=True, inplace=True)
        dat_out.rio.to_raster(raster_path=os.path.join(output_folder, 'g_2_min_id', out_name), driver='COG')
        
        # Group 1 Band Depth
        print("  Saving Group 1 Band Depth...")
        out_name = f'{ds_min.granule_id}_group_1_band_depth.tif'
        dat_out = ds_min['group_1_band_depth']
        dat_out.data = np.nan_to_num(dat_out.data, nan=-9999)
        dat_out.rio.write_nodata(-9999, encoded=True, inplace=True)
        dat_out.rio.to_raster(raster_path=os.path.join(output_folder, 'g_1_band_d', out_name), driver='COG')
        
        # Group 2 Band Depth
        print("  Saving Group 2 Band Depth...")
        out_name = f'{ds_min.granule_id}_group_2_band_depth.tif'
        dat_out = ds_min['group_2_band_depth']
        dat_out.data = np.nan_to_num(dat_out.data, nan=-9999)
        dat_out.rio.write_nodata(-9999, encoded=True, inplace=True)
        dat_out.rio.to_raster(raster_path=os.path.join(output_folder, 'g_2_band_d', out_name), driver='COG')
    
    print(f"✓ Completed processing: {os.path.basename(fp)}")

print(f"\n{'='*60}")
print("All files processed successfully!")
print(f"Output directory: {output_folder}")
print(f"Total files processed: {len(nc_files_list)}")
print(f"{'='*60}")
