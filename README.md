# RTStruct to Numpy

## Installation

python setup.py install

## Usage

Define a path containing the RTStruct and the corresponding Dicom Images

```python
import rts_to_npy

study = rts_to_npy.load(dicom_path)
```

By default only contours with less than 50000 coordinates will be loaded. This can be changed using the optional parameter *max_number_of_coords*.

The function returns an object containing the image with all VOIs.

```python
study = rts_to_npy.load(dicom_path)

print(study) # List VOIs by name

study.image	# Returns image as numpy

study.Cerebellum.npy() # Returns mask of VOI named Cerebellum as numpy

study.Cerebellum.plot() # Plots image with contour (slice and zoom optional)
```





