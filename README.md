# MegaStream

Full pipeline for CAD-based generalizable 6DoF pose estimation, based on CNOS and MegaPose.

## Modules
*Please check the original page for more information.*
- [CNOS](https://github.com/nv-nguyen/cnos)
- [MegaPose](https://github.com/megapose6d/megapose6d)

## Installation 
1. Clone the repository
```
git clone https://github.com/Jianxff/megastream
cd megastream
```

2. Setup conda environment
```
conda env create -f environment.yml
conda activate megastream
```

3. Download checkpoints [Optional]
```
# pre-download checkpoints for CNOS and MegaPose
# the checkpoints will also be automatically downloaded when running the pipeline
python -m scripts.download_checkpoints
```

## Todo
- [x] Integrate MobileSAM for faster 2D detection
- [x] Integrate GigaPose for faster coarse estimation

## Note
- The Pipelien is currently support for `single object` detection and pose estimation.
- The modified CNOS model is based on `FastSAM` , for other model such as SAM and MobileSAM, edit the code at `${PROJECT_ROOT}/modules/cnos`.
- Due to render conflict or other reason, the template-rendering process is not able to be integrated into full pipeline. 
- The process speed is related to the resolution of input frames and the size of object model.

## Usage
1. Pre-Process CAD model
```
# only *.obj and *.ply are supported
python -m scripts.register_object ${path/to/model.ply(obj)}
```

2. Run the pipeline on video
```
# Run basic pipeline on rgb video
python -m scripts.run_megastream_on_video 
    --input ${path/to/rgb.mp4} 
    --object ${path/to/model.ply(obj)} 
    --output ${path/to/output.mp4}
```

- For using depth map to run ipc refine
```
# Depth images should be packed as RGB video
# Type '--depth ${path/to/depth.mp4}' as argument
```

- For simulating real-time process of input frames
```
# Type '--no-sync' as argument
```

## Test on HO3D Dataset
**Synchronized Process**
<video src="https://github.com/Jianxff/megastream/assets/35654252/5089aa68-8739-42ad-b2e7-1ba178e25ea6"></video>

**Realtime Simulation**
<video src="https://github.com/Jianxff/megastream/assets/35654252/c93d955d-c5b1-4e4a-beb8-823e858aae3a"></video>

