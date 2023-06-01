# Art Attack 🎨

This project provides users with an easy-to-use and fun way to draw images virtually, eliminating the need for traditional brush and color setup. It offers a wide range of tools to edit and enhance created art.

## Features

1. *Virtual Painter*: Users can draw in a virtual environment by waving their hand in the air. The camera captures hand movements, displaying the drawing on the camera screen and a separate canvas. Users can select shapes and colors using a GUI, controlled by finger gestures.

2. *Background Remover*: Allows users to remove the background of any image from their system. The background is replaced with a white sheet, and the modified image can be saved.

3. *Background Changer*: Users can change the background of any selected image from their system with another image. The modified image can be saved.

4. *Artistic Effects*:
   - *Pointillistic Art*: Image divided into points using k-means clustering. Users can adjust radius and number of points for different effects.
   - *Sketching Effect*: Image rendered in grayscale resembling a pencil sketch. Both grayscale and color versions are saved.
   - *Watercolor Effect*: Transform image into a watercolor-like representation using cv2.stylization() function.
   - *Enhance Details*: Enhance sharpness and structure of an image using specific parameters.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- Tkinter
- Sklearn
- Scipy
- Numpy

## Installation

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the main script.

## Usage

1. Launch the application.
2. Choose the desired feature from the menu.
3. Follow the instructions provided by the GUI to interact with each feature.
4. Save the modified image as required.
5. Exit the application.

## License

This project is licensed under the [MIT License](LICENSE).
