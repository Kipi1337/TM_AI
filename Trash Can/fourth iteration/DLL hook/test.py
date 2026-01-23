import pygetwindow as gw
import mss
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np

def get_window_title():
    # Replace with the exact title of your Trackmania window
    return "Trackmania Modded Forever (2.12.0) [AI]: TMInterface (2.2.1), CoreMod (1.0.10)"

def capture_window_screenshot(window_title):
    # Get the window object by title
    window = gw.getWindowsWithTitle(window_title)
    if not window:
        raise ValueError(f"Window with title '{window_title}' not found.")
    window = window[0]

    # Get the window coordinates and size
    left, top, width, height = window.left, window.top, window.width, window.height

    print(f"Capturing window: {window_title}")
    print(f"Left: {left}, Top: {top}, Width: {width}, Height: {height}")

    # Create an mss instance
    with mss.mss() as sct:
        # Capture the screenshot of the window
        screenshot = sct.grab((left, top, left + width, top + height))

        # Convert to PIL Image
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)

    return img

def crop_image(image, crop_coords):
    # Crop the image based on the provided coordinates (left, upper, right, lower)
    cropped_image = image.crop(crop_coords)
    return cropped_image

def preprocess_image(image):
    # Convert to grayscale
    gray_image = image.convert('L')

    # Apply adaptive thresholding to convert text to black and white
    np_image = np.array(gray_image)
    _, thresh = cv2.threshold(np_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Sharpen the image using OpenCV
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]], np.float32)
    sharpened_image = cv2.filter2D(thresh, -1, sharpen_kernel)

    return Image.fromarray(sharpened_image)

def process_image(image, crop_coords):
    # Crop the image to focus on the overlay
    cropped_image = crop_image(image, crop_coords)

    # Save the cropped screenshot for inspection
    cropped_image.save('cropped_screenshot.png')

    # Preprocess the image
    processed_image = preprocess_image(cropped_image)
    processed_image.save('processed_screenshot.png')

    # Use OCR to extract text from the screenshot
    try:
        text = pytesseract.image_to_string(processed_image, lang='eng', config='--psm 6')
        print("OCR Success!")
        print("Extracted Text:")
        print(text)

        # Post-process the text to correct common misinterpretations
        corrected_text = text.replace('@', '0')
        print("Corrected Text:")
        print(corrected_text)
    except Exception as e:
        print(f"OCR Failed: {e}")

if __name__ == "__main__":
    window_title = get_window_title()
    screenshot = capture_window_screenshot(window_title)

    # Define the crop coordinates (left, upper, right, lower)
    crop_coords = (15, 50, 345, 235)  # Your specified coordinates

    process_image(screenshot, crop_coords)


# window_title = "Trackmania Modded Forever (2.12.0) [AI]: TMInterface (2.2.1), CoreMod (1.0.10)"  # Replace with the actual window title