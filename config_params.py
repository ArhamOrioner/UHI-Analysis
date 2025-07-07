import requests
import json

def get_city_roi_bounds(city_name):
    """
    Finds the approximate bounding box (ROI_BOUNDS) for a given city name
    using the OpenStreetMap Nominatim API.

    Args:
        city_name (str): The name of the city (e.g., "Mumbai", "New Delhi").

    Returns:
        list: A list of [west_longitude, south_latitude, east_longitude, north_latitude]
              if the city is found, otherwise None.
    """
    # Nominatim API endpoint for searching
    url = "https://nominatim.openstreetmap.org/search"
    
    # Parameters for the API request
    params = {
        'q': city_name,         # The query string (city name)
        'format': 'jsonv2',     # Request JSON output (version 2 for better structure)
        'limit': 1              # Limit to 1 result (usually the most relevant)
    }
    
    # It's good practice to set a User-Agent for Nominatim requests
    # Replace 'YourApplicationName/1.0' with something descriptive if you use this in a project.
    headers = {
        'User-Agent': 'UHI_Analysis_Script/1.0 (your_email@example.com)'
    }

    print(f"Searching for '{city_name}' using Nominatim API...")
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        
        data = response.json()

        if not data:
            print(f"No results found for '{city_name}'. Please try a different name or spelling.")
            return None
        
        # The first result is usually the most relevant
        first_result = data[0]
        
        # Nominatim boundingbox format: [latitude_south, latitude_north, longitude_west, longitude_east]
        # We need: [longitude_west, latitude_south, longitude_east, latitude_north] for ee.Geometry.Rectangle
        bbox = first_result['boundingbox']
        
        # Convert string coordinates to float and reorder
        south_lat = float(bbox[0])
        north_lat = float(bbox[1])
        west_lon = float(bbox[2])
        east_lon = float(bbox[3])
        
        roi_bounds = [west_lon, south_lat, east_lon, north_lat]
        
        print(f"\nFound bounds for '{city_name}':")
        print(f"  West Longitude: {west_lon}")
        print(f"  South Latitude: {south_lat}")
        print(f"  East Longitude: {east_lon}")
        print(f"  North Latitude: {north_lat}")
        
        return roi_bounds

    except requests.exceptions.RequestException as e:
        print(f"Network error or API request failed: {e}")
        print("Please check your internet connection or try again later.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON response from API. Invalid response format.")
        return None
    except KeyError:
        print(f"Could not extract bounding box from API response for '{city_name}'. Response might be incomplete.")
        return None
    except ValueError as e:
        print(f"Error converting coordinates to numbers: {e}")
        return None

if __name__ == "__main__":
    city = input("Enter city name (e.g., New Delhi, Mumbai, Tokyo): ")
    
    bounds = get_city_roi_bounds(city)
    
    if bounds:
        print("\n------------------------------------------------------------")
        print("Copy the following line and paste it into your uhi_analysis.py script:")
        print(f"ROI_BOUNDS = {bounds}")
        print("------------------------------------------------------------")
        print("\nNote: This is an approximate bounding box for the city.")
        print("For more precise or custom study areas, use the Google Earth Engine Code Editor")
        print("drawing tools to define your own geometry and get its coordinates.")
