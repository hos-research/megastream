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

## Note
- The Pipelien is currently support for `single object` detection and pose estimation.
- The modified CNOS model is based on `FastSAM`, for other model such as SAM and MobileSAM, edit the code at `${PROJECT_ROOT}/modules/cnos`.
- Due to render conflict or other reason, the template-rendering process is not able to be integrated into full pipeline. 

## Usage
1. Pre-Process CAD model
```
# only *.obj and *.ply are supported
python -m scripts.register_object ${path/to/model.ply(obj)}
```

2. Run the pipeline on video
```
# for synchronized process
python -m scripts.run_megastream_on_video 
    --input ${path/to/input.mp4} 
    --object ${path/to/model.ply(obj)} 
    --output ${path/to/output.mp4}

# for simulated realtime process
python -m scripts.run_megastream_on_video 
    --input ${path/to/input.mp4} 
    --object ${path/to/model.ply(obj)} 
    --output ${path/to/output.mp4} 
    --realtime
```