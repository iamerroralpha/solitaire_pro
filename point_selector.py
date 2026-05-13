#!/usr/bin/env python3
"""
Point Selector - Click two points on your screen with your mouse.
The coordinates are saved and can be used for computer vision tasks.
"""

import cv2
import numpy as np
from PIL import ImageGrab
import json
from pathlib import Path


class PointSelector:
    def __init__(self):
        self.points = []
        self.image = None
        self.display_image = None
        self.window_name = "Point Selector - Click 2 points (ESC to cancel)"
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            
            # Draw circle on the clicked point
            cv2.circle(self.display_image, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(self.display_image, (x, y), 10, (0, 255, 0), 2)
            
            # Draw line between points if we have two
            if len(self.points) == 2:
                cv2.line(self.display_image, self.points[0], self.points[1], (255, 0, 0), 2)
            
            cv2.imshow(self.window_name, self.display_image)
    
    def select_points(self):
        """Main method to capture screen and select two points."""
        # Capture the entire screen
        print("\n🖼️  Capturing screen...")
        screenshot = ImageGrab.grab()
        self.image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        self.display_image = self.image.copy()
        
        # Display the image
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.display_image)
        
        print(f"Screen size: {self.image.shape[1]} x {self.image.shape[0]}\n")
        
        # Point 1
        print("━" * 40)
        print("👆 POINT 1: Click on the first location")
        print("━" * 40)
        while len(self.points) < 1:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("\n❌ Cancelled by user.")
                cv2.destroyAllWindows()
                return None
        print(f"✓ Point 1 selected: {self.points[0]}\n")
        
        # Point 2
        print("━" * 40)
        print("👆 POINT 2: Click on the second location")
        print("━" * 40)
        while len(self.points) < 2:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("\n❌ Cancelled by user.")
                cv2.destroyAllWindows()
                return None
        print(f"✓ Point 2 selected: {self.points[1]}\n")
        
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        return self.points
    
    def save_points(self, filename="selected_points.json"):
        """Save points to a JSON file."""
        if len(self.points) != 2:
            print("Error: Need exactly 2 points to save.")
            return False
        
        data = {
            "point1": {"x": self.points[0][0], "y": self.points[0][1]},
            "point2": {"x": self.points[1][0], "y": self.points[1][1]},
            "points_tuple": self.points
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Points saved to {filename}")
        return True


def main():
    print("\n" + "="*40)
    print("  POINT SELECTOR")
    print("="*40)
    
    selector = PointSelector()
    points = selector.select_points()
    
    if points:
        print("="*40)
        print("✅ SUCCESS!")
        print("="*40)
        print(f"Point 1: {points[0]}")
        print(f"Point 2: {points[1]}")
        
        # Save to file
        selector.save_points("selected_points.json")
        print("\n✓ Points saved to selected_points.json")
    else:
        print("\n❌ No points selected.")


if __name__ == "__main__":
    main()
