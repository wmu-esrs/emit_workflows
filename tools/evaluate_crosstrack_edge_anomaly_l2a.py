
import os
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import traceback
import yaml


# === CONFIGURATION ===
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

input_folder = config["l2a"]
output_folder = config["output_folder_l2a_edge_anomaly_plots"]


os.makedirs(output_folder, exist_ok=True)

# === SETUP ===
warnings.simplefilter('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# === EDGE DIFFERENCE FUNCTION ===
def compute_edge_diff(neighbor_pixels, reference_pixels):
    # neighbor pixels from 0 to 9, 0 being the edge pixels and reference being the 10th pixel
    return np.nanmean(neighbor_pixels - reference_pixels, axis=0)

def process_scene(file_path):
    try:
        ds = xr.open_dataset(file_path)
        wvl = xr.open_dataset(file_path, group='sensor_band_parameters')
        loc = xr.open_dataset(file_path, group='location')

        ds = ds.assign_coords({
            'downtrack': ds.downtrack.data,
            'crosstrack': ds.crosstrack.data,
            **wvl.variables,
            **loc.variables
        }).swap_dims({'bands': 'wavelengths'})

        reflectance = ds['reflectance']
        n_downtrack, n_crosstrack = reflectance.sizes['downtrack'], reflectance.sizes['crosstrack']

        edges = {
            'top': [reflectance.isel(downtrack=i).values for i in range(11)],
            'bottom': [reflectance.isel(downtrack=n_downtrack - 1 - i).values for i in range(11)],
            'left': [reflectance.isel(crosstrack=i).values for i in range(11)],
            'right': [reflectance.isel(crosstrack=n_crosstrack - 1 - i).values for i in range(11)],
        }

        diffs = {k: [compute_edge_diff(v[i], v[10]) for i in range(10)] for k, v in edges.items()}
        wavelengths = reflectance.wavelengths.values
        return diffs, wavelengths, {k: v[0].shape[0] for k, v in edges.items()}

    except Exception as e:
        logging.error(f"Failed to process {os.path.basename(file_path)}: {e}")
        traceback.print_exc()
        return None, None, None

def plot_diffs(diffs, wavelengths, pixel_counts, file_name):
    colors = {
        'top': 'red',
        'bottom': 'blue',
        'left': 'green',
        'right': 'purple'
    }

    plt.figure(figsize=(10, 6))
    for edge, base_color in colors.items():
        for i in range(10):
            plt.plot(
                wavelengths,
                diffs[edge][i],
                #label=f'{edge.capitalize()} Edge - {i}th neighbor minus 10th',
                label = f'{edge.capitalize()} Edge - {i}th neighbor minus 10th (px n:{pixel_counts[edge]})',
                color=base_color,
                alpha= 1-((i + 1) / 10)
            )

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Average Reflectance Difference')
    plt.title(f'Spectral Difference between Edge Pixels and the 10th Neighbor \n {file_name}')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, prop={'size': 4})
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(output_folder, f"{file_name}_edge_diff.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    top_diffs, bottom_diffs, left_diffs, right_diffs = [], [], [], []
    wavelengths = None

    total_pixels = {"top": 0,
                    "bottom": 0,
                    "left": 0,
                    "right": 0
                   }
    # total_top_pixels = 0
    # total_bottom_pixels = 0
    # total_left_pixels = 0
    # total_right_pixels = 0
    
    scene_count = 0

    for file_name in tqdm(os.listdir(input_folder)):
        if not file_name.endswith(".nc"):
            continue

        scene_path = os.path.join(input_folder, file_name)
        logging.info(f"Processing {file_name}...")

        diffs, wvl, pixel_counts = process_scene(scene_path)
        if diffs is None:
            continue

        top_diffs.append(diffs['top'])
        bottom_diffs.append(diffs['bottom'])
        left_diffs.append(diffs['left'])
        right_diffs.append(diffs['right'])

        wavelengths = wvl  # Set only once

        
        total_pixels['top'] += pixel_counts['top']
        total_pixels['bottom'] += pixel_counts['bottom']
        total_pixels['left'] += pixel_counts['left']
        total_pixels['right'] += pixel_counts['right']

        # total_top_pixels += pixel_counts['top']
        # total_bottom_pixels += pixel_counts['bottom']
        # total_left_pixels += pixel_counts['left']
        # total_right_pixels += pixel_counts['right']
        
        scene_count += 1

        plot_diffs(diffs, wavelengths, pixel_counts, file_name)

    # === AGGREGATE PLOT ===
    if wavelengths is not None and len(top_diffs) > 0:
        top_avg = np.nanmean(np.stack(top_diffs), axis=0)
        bottom_avg = np.nanmean(np.stack(bottom_diffs), axis=0)
        left_avg = np.nanmean(np.stack(left_diffs), axis=0)
        right_avg = np.nanmean(np.stack(right_diffs), axis=0)

        plt.figure(figsize=(10, 6))
        colors = {
            'top': 'red',
            'bottom': 'blue',
            'left': 'green',
            'right': 'purple'
        }
        for edge, base_color in colors.items():
            for i in range(10):
                plt.plot(
                    wavelengths,
                    diffs[edge][i],
                    label = f'{edge.capitalize()} Edge - {i}th neighbor minus 10th (px n:{total_pixels[edge]})',
                    color=base_color,
                    alpha= 1-((i + 1) / 10)
                )

        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Average Reflectance Difference')
        plt.title(f'Spectral Difference between Edge Pixels and the 10th Neighbor across {scene_count} Scenes')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, prop={'size': 4})
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(output_folder, "edge_anomaly_of_all_scenes.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved aggregate plot to {output_path}")

       
if __name__ == "__main__":
    main()
