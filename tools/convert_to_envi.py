# This script converts image cubes to ENVI format and orthorectifies them in parallel batches.
# It automatically exits if RAM usage exceeds 95% 
# After conversion, it renames ENVI files by removing the '_reflectance' extension from both .img and .hdr files.
# The script also removes the content of the 'description' field from .hdr files to ensure compatibility with tetracorder

import os
import sys
import logging
import traceback
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import psutil
import csv
import atexit
import emit_tools as et
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import re

# sys.path.append('../modules/')
# import emit_tools as et


# === CONFIGURATION ===
print("Loading configuration from config.yaml...")
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

input_folder = config["l2a_edge_masked"]
output_folder = config["l2a_envi"]
log_file = config["log_convert_to_envi"]
log_resource_use = config["log_convert_to_envi_resource_use"]

# make sure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Logging Setup ---
																
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# --- CSV Setup for resource logging ---
resource_data = []
							   
						

def log_to_csv(cpu, mem, disk, tag):
    resource_data.append({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "cpu_percent": cpu,
        "mem_percent": mem.percent,
        "mem_used_mb": mem.used // (1024**2),
        "mem_total_mb": mem.total // (1024**2),
        "disk_percent": disk.percent,
        "disk_free_gb": disk.free // (1024**3)
    })

# --- Exit Hook to Dump CSV + Plot ---
@atexit.register
def dump_csv_and_plot():
    if not resource_data:
        return

    df = pd.DataFrame(resource_data)
    df.to_csv(log_resource_use, index=False)
    logging.info(f"System resource usage written to {log_resource_use}")
    print(f"\nCSV saved: {log_resource_use}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["timestamp"], df["cpu_percent"], label="CPU %", color='blue')
    plt.plot(df["timestamp"], df["mem_percent"], label="RAM %", color='red')
    plt.xticks(rotation=45)
    plt.ylabel("Usage (%)")
    plt.xlabel("Time")
    plt.title("System Resource Usage During Processing")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_folder, "resource_usage_plot.png")
    plt.savefig(plot_path)
    logging.info(f"Resource usage plot saved to {plot_path}")
    print(f"Plot saved: {plot_path}")

# --- File listing ---
def list_nc_files(directory):
    nc_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nc'):
                nc_files.append(os.path.join(root, file))
    return nc_files

# --- Monitor system ---
def log_system_status(tag=""):
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.5)
    disk = psutil.disk_usage(output_folder)
    log_to_csv(cpu, mem, disk, tag)

    status_msg = (
        f"{tag}System Status: CPU {cpu}%, RAM {mem.percent}% used, Disk {disk.percent}% used"
    )

    logging.info(status_msg)
    print(status_msg)

    # Abort if memory usage is dangerously high
    if mem.percent > 95:
        alert_msg = f"CRITICAL: Memory usage exceeded 95% ({mem.percent}%) — aborting for system safety."
        logging.critical(alert_msg)
        print(alert_msg)
        raise SystemExit(alert_msg)

# --- Main processing function ---
def process_nc_file(fp):
    base_filename = os.path.basename(fp).replace('.nc', '')
    target_img = os.path.join(output_folder, base_filename)
    target_hdr = target_img.replace('.img', '.hdr')

    ## disabled the existence check to overwrite existing files
    # if os.path.exists(target_img) and os.path.exists(target_hdr):
    #     msg = f"Skipped (already exists): {fp}"
    #     logging.info(msg)
    #     print(msg)
    #     return msg

    try:
        log_system_status(f"[{os.path.basename(fp)}] ")
        logging.info(f"Processing: {fp}")
        print(f"Processing: {fp}")

        ds = et.emit_xarray(fp, ortho=True)
        et.write_envi(ds, output_folder, overwrite=False, extension='.img', interleave='BIL', glt_file=False)

        msg = f"Processed: {fp}"
        logging.info(msg)
        print(msg)
        return msg
    except Exception as e:
        tb = traceback.format_exc()
        msg = f"Error processing {fp}: {e}\n{tb}"
        logging.error(msg)
        print(msg)
        return msg
    
def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".img") or filename.endswith(".hdr"):
            new_filename = filename.replace("_reflectance.img", "").replace("_reflectance", "")
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")
    
def modify_hdr_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".hdr"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                content = file.read()
            
            modified_content = re.sub(r'description\s*=\s*\{.*?\}', 'description = {}', content, flags=re.DOTALL)
            
            with open(file_path, "w") as file:
                file.write(modified_content)
            print(f"Modified: {filename}")

# --- Execute ---
if __name__ == '__main__':
    nc_files_list = list_nc_files(input_folder)
    max_workers = min(5, os.cpu_count()) # make changes number of image cubes to convert to ENVi in parallel(here 5)

    logging.info(f"Found {len(nc_files_list)} files to process with {max_workers} threads.")
    print(f"Processing {len(nc_files_list)} files with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_nc_file, fp): fp for fp in nc_files_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            result = future.result()
            print(result)
    
    # rename envi and hdr files and remove description from HDR files
    rename_files(output_folder)
    modify_hdr_files(output_folder)
