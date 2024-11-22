import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess the Image
import cv2

def load_and_preprocess(image_path):
    """
    Load the image in grayscale and binarize it.
    Black text should appear on a white background.
    """
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Binarize the image (invert to ensure text is black and background is white)
    _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    
    # Show the binary image in a window
    cv2.imshow("Binarized Image", binary_image)
    
    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return binary_image


# Step 2: Identify Empty Rows
def find_empty_rows(binary_image):
    """
    Find rows in the binary image that are completely empty (all pixels are white).
    Group consecutive empty rows into intervals.
    """
    # Each row is empty if all its pixels are zero in the inverted binary image (255 - black background)
    empty_rows = np.all(binary_image == 255, axis=1)
    
    # Identify regions of consecutive empty rows and return them as intervals
    empty_regions = []
    start = None
    for i, is_empty in enumerate(empty_rows):
        if is_empty and start is None:
            start = i  # Start of a new empty region
        elif not is_empty and start is not None:
            empty_regions.append((start, i - 1))  # End of the current empty region
            start = None
    # If the last rows are empty, add the final region
    if start is not None:
        empty_regions.append((start, len(empty_rows) - 1))
    
    return empty_regions

# Step 3: Draw Horizontal Lines
def draw_horizontal_lines(image, empty_regions):
    """
    Draw horizontal lines on the empty regions of the image.
    Each region will have a unique color.
    """
    # Convert binary image to BGR for color visualization
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Assign unique colors for each region
    for i, (start, end) in enumerate(empty_regions):
        color = tuple(np.random.randint(0, 256, size=3).tolist())  # Random color
        # Draw a line in the middle of each empty region
        mid_row = (start + end) // 2
        cv2.line(color_image, (0, mid_row), (color_image.shape[1], mid_row), color, thickness=2)
    
    return color_image

# Main Script
if __name__ == "__main__":
    # Replace with the path to your uploaded resume image
    image_path = "C:/Users/hp/Desktop/nicheby/myResume.png"  # Update the path as needed

    # Step 1: Load and preprocess the image
    binary_image = load_and_preprocess(image_path)

    # Step 2: Find empty rows
    empty_regions = find_empty_rows(binary_image)
    print("Empty Regions:", empty_regions)

    # Step 3: Draw horizontal lines
    colorized_image = draw_horizontal_lines(binary_image, empty_regions)

    # Display the original and processed image
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.title("Binary Image")
    plt.imshow(binary_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Image with Horizontal Lines")
    plt.imshow(cv2.cvtColor(colorized_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Save the result to a file
    output_path = "Processed_Resume.jpg"
    cv2.imwrite(output_path, colorized_image)
    print(f"Processed image saved to {output_path}")
