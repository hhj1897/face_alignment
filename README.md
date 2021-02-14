# ibug.face_alignment
TODO

## Prerequisites
* [Numpy](https://www.numpy.org/): `$pip3 install numpy`
* [OpenCV](https://opencv.org/): `$pip3 install opencv-python`
* [PyTorch](https://pytorch.org/): `$pip3 install torch torchvision`
* [ibug.face_detection](https://github.com/hhj1897/face_detection) (only needed by the test script): See this repository for details: [https://github.com/hhj1897/face_detection](https://github.com/hhj1897/face_detection).

## How to Install
```
git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
```

## How to Test
* To test on live video: `python face_alignment_test.py [-i webcam_index]`
* To test on a video file: `python face_alignment_test.py [-i input_file] [-o output_file]`

## How to Use
TODO

## References
\[1\] Bulat, Adrian, and Georgios Tzimiropoulos. "[How far are we from solving the 2d & 3d face alignment problem?(and a dataset of 230,000 3d facial landmarks).](http://openaccess.thecvf.com/content_ICCV_2017/papers/Bulat_How_Far_Are_ICCV_2017_paper.pdf)" In _Proceedings of the IEEE International Conference on Computer Vision_, pp. 1021-1030. 2017.
