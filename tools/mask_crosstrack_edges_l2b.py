import os
import csv
import time
import numpy as np
from netCDF4 import Dataset
from multiprocessing import Pool, cpu_count
import yaml

# === CONFIGURATION ===
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

input_folder = config["l2b"]
output_folder = config["l2b_edge_masked"]
log_csv_path = config["log_csv_path_l2b"]

# make sure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def mask_crosstrack_edges_l2b(input_output_tuple):
    input_path, output_path, edge_width, max_retries = input_output_tuple
    attempt = 0
    success = False
    error_msg = ""

    vars_to_mask = [
        "group_1_band_depth",
        "group_1_mineral_id",
        "group_2_band_depth",
        "group_2_mineral_id"
    ]

    print(f"==> Starting: {input_path}")
    print(f"    Output: {output_path}")
    print(f"    Edge width: {edge_width}")

    # Ensure output directory exists
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except Exception as e:
        print(f"    ERROR: Failed to create output directory: {e}")
        return {
            "filename": os.path.basename(input_path),
            "status": "FAILED (output directory creation failed)",
            "error": str(e),
            "retries": 0
        }

    while attempt < max_retries and not success:
        try:
            with Dataset(input_path, "r") as src:
                with Dataset(output_path, "w", format="NETCDF4") as dst:
                    # Copy dimensions
                    for name, dim in src.dimensions.items():
                        dst.createDimension(name, len(dim) if not dim.isunlimited() else None)

                    # Copy global attributes
                    dst.setncatts({attr: src.getncattr(attr) for attr in src.ncattrs()})
                    history = src.getncattr("history") if "history" in src.ncattrs() else ""
                    dst.setncattr("history", history + "; cross-track edges masked with NoData")

                    # Copy variables in root group, preserving chunk sizes
                    for name, varin in src.variables.items():
                        print(f"    Copying variable: {name}")
                        fill_value = varin.getncattr('_FillValue') if '_FillValue' in varin.ncattrs() else None
                        # Get chunking info
                        chunksizes = varin.chunking() if hasattr(varin, "chunking") and varin.chunking() != 'contiguous' else None
                        if fill_value is not None:
                            varout = dst.createVariable(name, varin.datatype, varin.dimensions, fill_value=fill_value, chunksizes=chunksizes)
                            attrs = {k: varin.getncattr(k) for k in varin.ncattrs() if k != '_FillValue'}
                        else:
                            varout = dst.createVariable(name, varin.datatype, varin.dimensions, chunksizes=chunksizes)
                            attrs = {k: varin.getncattr(k) for k in varin.ncattrs()}
                        varout.setncatts(attrs)
                        data = varin[:]

                        if name in vars_to_mask and fill_value is not None:
                            print(f"    Masking edges in {name} with fill value {fill_value}")
                            if data.ndim == 2:
                                data[:, :edge_width] = fill_value
                                data[:, -edge_width:] = fill_value
                            else:
                                print(f"    Skipping edge masking for {name}, unexpected shape: {data.shape}")
                        elif name in vars_to_mask:
                            print(f"    WARNING: No _FillValue found for variable {name}, skipping masking.")

                        varout[:] = data

                    # Recursively copy groups, preserving chunk sizes
                    def copy_group(src_group, dst_group):
                        for name, dim in src_group.dimensions.items():
                            if name not in dst_group.dimensions:
                                dst_group.createDimension(name, len(dim) if not dim.isunlimited() else None)

                        for name, varin in src_group.variables.items():
                            print(f"    Copying group variable: {name}")
                            fill_value = varin.getncattr('_FillValue') if '_FillValue' in varin.ncattrs() else None
                            chunksizes = varin.chunking() if hasattr(varin, "chunking") and varin.chunking() != 'contiguous' else None
                            if fill_value is not None:
                                varout = dst_group.createVariable(name, varin.datatype, varin.dimensions, fill_value=fill_value, chunksizes=chunksizes)
                                attrs = {k: varin.getncattr(k) for k in varin.ncattrs() if k != '_FillValue'}
                            else:
                                varout = dst_group.createVariable(name, varin.datatype, varin.dimensions, chunksizes=chunksizes)
                                attrs = {k: varin.getncattr(k) for k in varin.ncattrs()}
                            varout.setncatts(attrs)
                            data = varin[:]
                            if name in vars_to_mask and fill_value is not None:
                                print(f"    Masking edges in {name} with fill value {fill_value}")
                                if data.ndim == 2:
                                    data[:, :edge_width] = fill_value
                                    data[:, -edge_width:] = fill_value
                                else:
                                    print(f"    Skipping edge masking for {name}, unexpected shape: {data.shape}")
                            elif name in vars_to_mask:
                                print(f"    WARNING: No _FillValue found for variable {name}, skipping masking.")
                            varout[:] = data

                        dst_group.setncatts({attr: src_group.getncattr(attr) for attr in src_group.ncattrs()})

                        for name, group in src_group.groups.items():
                            print(f"    Entering subgroup: {name}")
                            copy_group(group, dst_group.createGroup(name))

                    for name, group in src.groups.items():
                        print(f"  Copying group: {name}")
                        copy_group(group, dst.createGroup(name))

            print(f"[✓] {os.path.basename(input_path)} complete.")
            success = True

        except Exception as e:
            attempt += 1
            error_msg = str(e)
            print(f"[✗] Attempt {attempt} failed for {os.path.basename(input_path)}: {error_msg}")
            time.sleep(1)

    return {
        "filename": os.path.basename(input_path),
        "status": "DONE" if success else f"FAILED (after {max_retries} attempts)",
        "error": "" if success else error_msg,
        "retries": attempt
    }


def process_all_scenes(input_folder, output_folder, log_csv_path, edge_width=1, num_workers=None, max_retries=3):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)

    task_list = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".nc"):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                task_list.append((input_path, output_path, edge_width, max_retries))

    print(f"\nStarting processing with {num_workers or cpu_count()} workers...")
    print(f"Total NetCDF files: {len(task_list)}\n")

    with Pool(processes=num_workers or cpu_count()) as pool:
        results = pool.map(mask_crosstrack_edges_l2b, task_list)

    # Write results to CSV
    with open(log_csv_path, "w", newline="") as csvfile:
        fieldnames = ["filename", "status", "retries", "error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"\nLog written to: {log_csv_path}")

# --- Execute ---
if __name__ == '__main__':
    process_all_scenes(
        input_folder=input_folder,
        output_folder=output_folder,
        log_csv_path=log_csv_path,
        edge_width=config.get("edge_width", 1),
        num_workers=config.get("num_workers", None),
        max_retries=config.get("max_retries", 3)
    )

