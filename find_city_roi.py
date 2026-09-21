"""Interactive helper using the same geocoder as the analysis CLI."""

from utils import get_city_roi_bounds

if __name__ == "__main__":
    bounds = get_city_roi_bounds(input("City name: ").strip())
    if bounds:
        print("--roi-bounds " + " ".join(str(v) for v in bounds))
        print("This is an approximate bounding box; review it for your study area.")
    else:
        raise SystemExit("City not found; supply a custom ROI.")
