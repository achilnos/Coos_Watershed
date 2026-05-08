# rain_field.py

import requests
import numpy as np
from PIL import Image
import io

def get_rain_grid_final_v2( lat, lon) :
    base_url = "https://geo.weather.gc.ca/geomet"
    layer = "RADAR_1KM_RRAI"
    # Use the timestamp your previous run successfully identified
    time_str = "2026-03-22T07:42:00Z"    
    #bbox = f"{lon-0.1},{lat-0.1},{lon+0.1},{lat+0.1}"# BBOX: minLon, minLat, maxLon, maxLat plus and minus buffer
    distance = .4
    bbox = f"{lon-distance},{lat-distance},{lon+0.1},{lat+distance}"
    
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "SRS": "EPSG:4326", # oddly, WMS 1.1.1 uses 'SRS' instead of 'CRS'
        "BBOX": bbox,
        "WIDTH": "100",
        "HEIGHT": "100",
        "FORMAT": "image/png",
        #"TIME": time_str,# I can't get it working right with the time paramter, but at least it give current data when I tested it. 
        "STYLES": "",
        "TRANSPARENT": "TRUE"
    }

    print(f"Requesting PNG grid via WMS 1.1.1...")
    response = requests.get( base_url, params = params, timeout = 20 )

    if response.status_code == 200:
        content_type = response.headers.get( "Content-Type", "" )
        if "image/png" in content_type:
            img = Image.open(io.BytesIO(response.content)).convert( 'L' )# 'L' converts to greyscale
            img_array = np.array( img ).astype( float )# prepare array to go back to float type
            rain_grid = img_array * 0.1# assume scaling of (0.1 mm/h per pixel unit)
            rain_grid[ img_array >= 254 ] = 0.0# set 'no data' values which are 255 to zero instead
            return rain_grid, time_str, distance
            # currently working here
            
        else:
            print("--- SERVER ERROR TEXT ---")
            print(response.text) # Error messages from server prints here
            return None, "Server Error"

    return None, f"HTTP {response.status_code}"


# I need to test again to see if it worked, this time with a map to display the result.
# i also need to resolve the bug related to the time_str variable not working anymore

# Montreal = 45.5019 , -73.5674
# Toronto = 43.6548, -79.3884
# New York City = 40.7128, -74.0060
# Seattle = 47.6062 , -122.3321
# Coos Bay = 43.3672, -124.2131
lat, lon = 43.3672, -124.2131
grid, info, distance = get_rain_grid_final_v2( lat, lon )


if grid is not None:
    print(f"Rain grid retrieved! Max value in current view: {np.max(grid)} mm/h")

else:
    print(f"FAILED: {info}")

# display the results here:
from test_rain_disp import display_rain_map, display_rain_with_context

#display_rain_map( grid, info )
display_rain_with_context( grid , distance , lat, lon , info )