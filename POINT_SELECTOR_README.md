# Point Selector - Quick Start Guide

A simple Python tool to select two points on your screen using your mouse, perfect for computer vision tasks.

## Setup

Install required dependencies:
```bash
pip install opencv-python pillow numpy
```

## Usage

### Step 1: Run the Point Selector
```bash
python point_selector.py
```

This will:
1. Capture your entire screen
2. Display it in a window
3. Wait for you to click 2 points
4. Save the coordinates to `selected_points.json`

**Controls:**
- **Left Click** - Select a point (do this twice)
- **ESC** - Cancel

### Step 2: Use the Points in Your Code

The points are saved as JSON in `selected_points.json`:
```json
{
  "point1": {"x": 100, "y": 200},
  "point2": {"x": 300, "y": 400},
  "points_tuple": [[100, 200], [300, 400]]
}
```

### Example: Load and Process Points
```python
import json

with open('selected_points.json', 'r') as f:
    data = json.load(f)

p1 = (data['point1']['x'], data['point1']['y'])
p2 = (data['point2']['x'], data['point2']['y'])

# Use p1 and p2 in your computer vision code
```

### Run the Example Processing Script
```bash
python process_points.py
```

This demonstrates:
- Loading saved points
- Calculating distance between points
- Finding the midpoint
- Calculating the angle
- Visualizing on screen

## Integration with Your CV Code

Modify `process_points.py` or create your own script that:
1. Loads points from `selected_points.json`
2. Performs your computer vision tasks
3. Uses the point coordinates as needed

Enjoy! 🎨
