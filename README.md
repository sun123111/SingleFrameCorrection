# Single-Frame Correction
C++ Version of "Orthophoto Maps". Convert "Orthophoto Maps" from Python to C++.

It is a mapping software that generates personal maps (orthophotos) from drone images. You can only generate orthophotos of regions of interest using images (and sensory data).


# Installation
The project depends on proj, OpenCV, Eigen, and GDAL.
These dependencies can be installed directly via vcpkg:
```
vcpkg install proj:x64-windows
vcpkg install opencv:x64-windows
vcpkg install eigen3:x64-windows
vcpkg install gdal
```


# Execution
1. Input the POS file path  
   POS format: Photo Name	Longitude	Latitude	Absolute Altitude	Roll Angle	Pitch Angle	Yaw Angle
2. Parameters  
   Mainly including focal length, parameters, and ground altitude