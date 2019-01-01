# Structure from Motion - Implementation in Python

You can use the dataset from https://www.eth3d.net/datasets

### Dependencies
1. OpenCV
2. numpy
3. Scipy - from least_square optimisation during bundle adjustment
4. mayavi (if required for visualization)

Create a `data` --> `calibration` and `data` --> `images` folders.
Place the downloaded calibration file and images in the corresponding folder. Please check the `main.py` function for setting up the folders.

Run `python main.py`
