import os
import xarray as xr
import rioxarray
import pandas as pd
import numpy as np

# Country bounding boxes (longitude, latitude)
COUNTRY_BBOXES = {
    'DE': {'minx': 5.0, 'miny': 47.0, 'maxx': 15.0, 'maxy': 55.0},  # Germany
    'FR': {'minx': -5.0, 'miny': 41.0, 'maxx': 10.0, 'maxy': 51.0},  # France
    'IT': {'minx': 6.0, 'miny': 35.0, 'maxx': 18.0, 'maxy': 47.0},   # Italy
    'ES': {'minx': -10.0, 'miny': 35.0, 'maxx': 5.0, 'maxy': 44.0},  # Spain
    'GB': {'minx': -8.0, 'miny': 49.0, 'maxx': 2.0, 'maxy': 61.0},   # United Kingdom
    'PL': {'minx': 14.0, 'miny': 49.0, 'maxx': 24.0, 'maxy': 55.0},  # Poland
    'NL': {'minx': 3.0, 'miny': 50.0, 'maxx': 8.0, 'maxy': 54.0},    # Netherlands
    'BE': {'minx': 2.0, 'miny': 49.0, 'maxx': 6.0, 'maxy': 51.0},    # Belgium
    'AT': {'minx': 9.0, 'miny': 46.0, 'maxx': 17.0, 'maxy': 49.0},   # Austria
    'CH': {'minx': 5.0, 'miny': 45.0, 'maxx': 11.0, 'maxy': 48.0},   # Switzerland
    'CZ': {'minx': 12.0, 'miny': 48.0, 'maxx': 19.0, 'maxy': 51.0},  # Czech Republic
    'HU': {'minx': 16.0, 'miny': 45.0, 'maxx': 23.0, 'maxy': 49.0},  # Hungary
    'RO': {'minx': 20.0, 'miny': 43.0, 'maxx': 30.0, 'maxy': 48.0},  # Romania
    'BG': {'minx': 22.0, 'miny': 41.0, 'maxx': 29.0, 'maxy': 44.0},  # Bulgaria
    'GR': {'minx': 20.0, 'miny': 35.0, 'maxx': 28.0, 'maxy': 42.0},  # Greece
    'PT': {'minx': -10.0, 'miny': 36.0, 'maxx': -6.0, 'maxy': 42.0}, # Portugal
    'SE': {'minx': 11.0, 'miny': 55.0, 'maxx': 24.0, 'maxy': 69.0},  # Sweden
    'NO': {'minx': 4.0, 'miny': 58.0, 'maxx': 31.0, 'maxy': 71.0},   # Norway
    'FI': {'minx': 20.0, 'miny': 60.0, 'maxx': 31.0, 'maxy': 70.0},  # Finland
    'DK': {'minx': 8.0, 'miny': 54.0, 'maxx': 15.0, 'maxy': 58.0},   # Denmark
}

def generate_population_weighted_temperature_series(country_code, year, 
                                                 output_dir='output',
                                                 population_data_path='data/gpw_v4_population_count_rev11_2020_30_sec.tif',
                                                 pat="edh_pat_80efd24ab073ddb1b2c18ead2afec15c165bbbd1dd298d9af98947920b102ad632b9929bb9f0ec9435372079dfecbff4"):
    """
    Generate population-weighted temperature time series for a specific country and year.
    
    Parameters:
    -----------
    country_code : str
        ISO 3166-1 alpha-2 country code (e.g., 'DE' for Germany)
    year : int or str
        Year to analyze (e.g., 2020 or '2020')
    output_dir : str
        Directory to save output files
    population_data_path : str
        Path to the population TIF file
    pat : str
        Personal Access Token for EarthDataHub
        
    Returns:
    --------
    dict
        Dictionary containing results and statistics
    """
    
    # Validate country code
    country_code = country_code.upper()
    if country_code not in COUNTRY_BBOXES:
        available_countries = ', '.join(sorted(COUNTRY_BBOXES.keys()))
        raise ValueError(f"Country code '{country_code}' not supported. Available countries: {available_countries}")
    
    # Get country bounding box
    country_bbox = COUNTRY_BBOXES[country_code]
    
    print(f"Generating population-weighted temperature series for {country_code} in {year}")
    print(f"Bounding box: {country_bbox}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Load ERA5 Temperature Data ---
    
    print("Accessing ERA5 data from EarthDataHub...")
    
    try:
        # Use the user-provided method to open the Zarr dataset
        era5_single_levels = xr.open_dataset(
            f"https://edh:{pat}@data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
            chunks={},
            engine="zarr",
            storage_options={"client_kwargs": {"trust_env": True}},
        )

        # Select and process the 2m temperature variable
        t2m = era5_single_levels.t2m

        # Convert from Kelvin to Celsius
        t2m = t2m - 273.15
        t2m.attrs["units"] = "°C"
        t2m.attrs['long_name'] = '2m Temperature'

        # Adjust longitude coordinates from (0, 360) to (-180, 180)
        t2m = t2m.assign_coords(longitude=(((t2m.longitude + 180) % 360) - 180))
        # Roll the longitude dimension to center the data
        t2m = t2m.roll(longitude=int(len(t2m.longitude) / 2), roll_coords=True)

        # Rename dimensions to be compatible with rioxarray
        t2m = t2m.rename({'latitude': 'y', 'longitude': 'x'})
        t2m.rio.write_crs("epsg:4326", inplace=True)

        print("Successfully loaded and processed temperature data.")

    except Exception as e:
        print(f"An error occurred while accessing the ERA5 Zarr store: {e}")
        print("Please ensure your PAT is correct and you have the necessary libraries (zarr, fsspec) installed.")
        raise

    # --- 2. Load Population Data ---
    
    print("Loading population data...")
    
    if not os.path.exists(population_data_path):
        raise FileNotFoundError(f"Population data not found at {population_data_path}")
    
    print("Population data found. Loading...")
    population_2020 = rioxarray.open_rasterio(population_data_path, masked=True).squeeze()

    # --- 3. Select Year of Data and Calculate Weighted Average ---
    
    print(f"Selecting {year} data...")
    # Select the specified year data
    t2m_year = t2m.sel(valid_time=str(year))
    
    print("Clipping temperature and population data to country extent...")
    t2m_country = t2m_year.rio.clip_box(**country_bbox)
    pop_da_country = population_2020.rio.clip_box(**country_bbox)
    
    print("Re-projecting population data to match the temperature grid...")
    # Use reproject_match to align the population grid to the temperature grid
    pop_da_aligned = pop_da_country.rio.reproject_match(t2m_country)
    
    # Ensure no negative population values after interpolation and fill with 0
    pop_da_aligned = pop_da_aligned.where(pop_da_aligned > 0, 0)
    
    print("Calculating total population...")
    total_population = pop_da_aligned.sum()
    
    if total_population.item() == 0:
        raise ValueError("Total population in the selected area is zero. Cannot calculate weighted average.")
    
    print("Calculating population-weighted temperature...")
    # Multiply temperature by population weight at each grid cell
    weighted_temp = t2m_country * pop_da_aligned
    # Sum the weighted values across the spatial dimensions and divide by the total population
    pop_weighted_temp_timeseries = weighted_temp.sum(dim=['x', 'y']) / total_population

    # --- 4. Save Results to CSV ---
    
    print("Converting to pandas DataFrame and saving to CSV...")
    
    # Convert xarray to pandas DataFrame
    df = pop_weighted_temp_timeseries.to_dataframe(name='temperature_celsius')
    
    # Check what columns we actually have and handle the time column properly
    print(f"DataFrame columns: {list(df.columns)}")
    print(f"DataFrame index: {list(df.index.names)}")
    
    # Reset index to get all dimensions as columns
    df = df.reset_index()
    
    # Find the time column (it might be 'valid_time' or just 'time')
    time_col = None
    for col in df.columns:
        if 'time' in col.lower():
            time_col = col
            break
    
    if time_col is None:
        raise ValueError("Could not find time column in the DataFrame")
    
    print(f"Using time column: {time_col}")
    
    # Format the time column to date string
    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = df[time_col].dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    output_filename = f"{country_code.lower()}_population_weighted_temperature_{year}.csv"
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)
    print(f"Successfully saved temperature data to: {output_path}")
    
    # Calculate statistics
    stats = {
        'country_code': country_code,
        'year': year,
        'number_of_time_points': len(df),
        'date_range_start': df[time_col].min(),
        'date_range_end': df[time_col].max(),
        'mean_temperature': df['temperature_celsius'].mean(),
        'min_temperature': df['temperature_celsius'].min(),
        'max_temperature': df['temperature_celsius'].max(),
        'total_population': total_population.item(),
        'output_file': output_path
    }
    
    # Print statistics
    print(f"\nTemperature Statistics for {country_code} in {year}:")
    print(f"Number of time points: {stats['number_of_time_points']}")
    print(f"Date range: {stats['date_range_start']} to {stats['date_range_end']}")
    print(f"Mean temperature: {stats['mean_temperature']:.2f}°C")
    print(f"Min temperature: {stats['min_temperature']:.2f}°C")
    print(f"Max temperature: {stats['max_temperature']:.2f}°C")
    print(f"Total population: {stats['total_population']:,.0f}")
    
    print(f"\nData saved successfully! You can now open {output_path} in Excel or any spreadsheet program.")
    
    return stats

def list_available_countries():
    """List all available country codes and their names."""
    country_names = {
        'DE': 'Germany', 'FR': 'France', 'IT': 'Italy', 'ES': 'Spain', 'GB': 'United Kingdom',
        'PL': 'Poland', 'NL': 'Netherlands', 'BE': 'Belgium', 'AT': 'Austria', 'CH': 'Switzerland',
        'CZ': 'Czech Republic', 'HU': 'Hungary', 'RO': 'Romania', 'BG': 'Bulgaria', 'GR': 'Greece',
        'PT': 'Portugal', 'SE': 'Sweden', 'NO': 'Norway', 'FI': 'Finland', 'DK': 'Denmark'
    }
    
    print("Available countries:")
    for code, name in sorted(country_names.items()):
        print(f"  {code}: {name}")
    
    return country_names

# Example usage
if __name__ == "__main__":
    # Example: Generate temperature series for Germany in 2020
    try:
        stats = generate_population_weighted_temperature_series('DE', 2020)
        print(f"\nFunction completed successfully!")
        print(f"Output file: {stats['output_file']}")
    except Exception as e:
        print(f"Error: {e}") 