# Pipeline RF-DETR-Preview-Seg to XML PAGE

## Description  

The purpose of this pipeline is to transform RF-DETR-Preview-Seg predictions files in the Coco format to XML Page files that can easily be imported in eScriptorium.

Leibniz's Manuscripts are characterized by complex layout, difficult to interpret by Kraken (the segmentation and HTR/OCR engine integrated in eScriptorium). The RF-DETR-Preview-Seg model is used specifically for the inference of text and graphic zones. We then run Kraken's baseline detection model and merge the resulting lines inside the detected zones. A third model, which is the HTR model can be applied inside the detected zones and lines.

## Limitations

There are certain limits to the launching of Kraken engine depending on the operating system.

Kraken dependancy limitations : 

On Mac with Intel chip (no GPU) :

- installable version : 5.2.9
- any higher version failed to install because of conflicts with compatible and installable torch versions
- Kraken 5.2.9 cannot be installed on Python 3.12 environment

On Mac with Silicon chip (M4) : 

- installable version (latest released at this date) : 7.0

Linux : 

- last version (7.0) should work, but not tested

Supervision dependancy limitations :

- only works on Python 3.12 environment
- for now, we run the inference via the API. It should work on all operating systems. Further tests will be performed on running the model locally.

We designed two ways of launching the code. One for Mac Intel, where two environments are automtically created and one for Linux and Mac Silicon, where the entire pipeline can be launched in one environment.

## Installation

### Mac Intel users : 

1. Install layout environment and dependencies :

```bash
conda env create -f environment_layout.yml
```

2. Install kraken environment and dependencies : 

```bash
conda env create -f environment_kraken.yml
```

3. Check that both environments have been installed : 

```bash
conda run -n layout_env python --version
```

```bash
conda run -n kraken_env python --version
```

4. Check Kraken version : 

```bash
conda run -n kraken_env kraken --version
```

### Linux and Mac Silicon users : 

1. Create virtual environment : 

```bash
conda create -y -n rfdetr2page_venv python=3.12
```

2. Activate the virtual environment : 

```bash
conda activate rfdetr2page_venv
```

3. Install dependencies : 

```bash
python -m pip install -r requirements.txt
```

## Usage

### Mac Intel users : 

1. Base command

```bash
bash run_pipeline.sh --input ./images --output ./results --config config.json
```

2. Base command + HTR (Handwritten Text Recognition)

```bash
bash run_pipeline.sh --input ./images --output ./results --config config.json --htr models/htr/FoNDUE-GD_v2_la.mlmodel
```

3. Extract zones (optional)

Same command + optional crop of the GraphicZone-figure zones (enabled by default), using the bounding box (bbox) shape (default):

```bash
bash run_pipeline.sh --input ./images --output ./results --config config.json --extract
```

4. Extract specific zones with polygon mode (optional) 

Same command + optional crop of one or more zones, using the polygon shape:

Each zone should be separated by a space after "--extract" parameter.

```bash
bash run_pipeline.sh --input ./images --output ./results --config config.json --extract MainZone --extract-mode polygon
```

### Linux and Mac Silicon users : 

1. Base command

```bash
python main_pipeline.py --input ./images --output ./results --config config.json
```

2. Base command + HTR (Handwritten Text Recognition)

```bash
python main_pipeline.py --input ./images --output ./results --config config.json --htr models/htr/FoNDUE-GD_v2_la.mlmodel
```

3. Extract zones (optional)

Same command + optional crop of the GraphicZone-figure zones (enabled by default), using the bounding box (bbox) shape (default):

```bash
python main_pipeline.py --input ./images --output ./results --config config.json --extract
```

4. Extract specific zones with polygon mode (optional) 

Same command + optional crop of one or more zones, using the polygon shape:

Each zone should be separated by a space after "--extract" parameter.

```bash
python main_pipeline.py --input ./images --output ./results --config config.json --extract MainZone --extract-mode polygon
```

To get all the parameters :

```bash
python main_pipeline.py --help
```

🔎 When we run the same command on the same files for the second time, the pipeline skips processing by default if the files have already been processed. To override results, add "-f" or "--force" to your command.

## Features

The pipeline is composed of the following steps : 

1. Inference on images present in the input folder
2. Conversion of COCO json format to XML-Page format
3. Creation of masks to hide the GraphicZone and eventually the Math Zones
4. Binarisation (this step was necessary for Kraken 5.2.9, but no longer necessary for Kraken 7.0)
5. Recreating images without the parts that were masked previously.
    * Since the "--mask" parameter is no longer proposed in Kraken 7.0, we recreate the images by completely erasing the zones that were previously masked inside the binary images before passing the baseline detection.
6. Baseline detection with Kraken (optionally with HTR/OCR depending on the user's usage)
7. Merging the detected lines with our exising zones (Kraken 5.2.9 would run region and baseline detection at the same time, erasing the original zones).
8. Simplifying the zone polygones for ease of correction : the initial polygons had many points, which made it difficult to correct the segmentation afterwards.
9. Fix regions by subtracting MarginTextZone zones from MainZone zones.
10. Fix lines spreading over multiple regions : cut them in pieces corresponding to the different regions and eliminate the rests (parts outside of any regions).
11. Extract images of certains regions (user's choice) in bounding box or/and polygon shape(s).

## Results

At the end you should find in your output folder the following subfolders : 

* 01_predictions_json
* 02_annotated_images
* 03_regions_xml
* 04_masks
* 05_binarized
* 05b_masked_for_baselines
* 06_baselines_and_htr
* 07_merged_xml
* 08_simplified_pagexml
* 09_fixed_region_overlaps
* 10_fixed_lines
* 11_extracted_regions (optional)
    - bbox
        - GraphicZone-figure
            - image_1_crop1.jpg
            - image_2_crop1.jpg
            - image_2_crop2.jpg
            ....
        - ...
    - polygon
        - GraphicZone-figure
        - ...

## Importing the results to eScriptorium

In order to import the final results to eScriptorium, the "10_fixed_lines" folder needs to be zipped. Then, the folder is ready to be imported.

⚠️ Make sure to use exactly the same image files as those you added in the "./images" folder, so that the segmentation matches the images. If the image dimensions change, the XML files do not correspond anymore.

## Author

Denisa-Florina Bumba, for the ERC Philiumm project

## Credits

Inspired by [YALTAi](https://github.com/PonteIneptique/YALTAi).
