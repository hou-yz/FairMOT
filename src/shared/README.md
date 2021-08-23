# Visualization tool for cattle annotation

## Requirements
```
opencv
numpy
matplotlib
```

## Code
`load_video.py` video loader.

`show_anno.py` visualization for tracking results. 

supported arguments:
+ `img_size`: custom image sizes for visualization (default: same resolution as in original video);
+ `step`: step for reading frames (read every X frames, default: 1 (read all frames))

## Data Preparation

specify the `video_fname` and `out_fpath` arguments as needed