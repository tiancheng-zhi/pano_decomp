# Application: Object Insertion


## Get started
You need to copy and paste transformations.py and utils.py from [zind](https://github.com/zillow/zind) to utils/
You can set up the environment with all dependencies like so:
```
pip install -r requirements.txt
```
You also need to install Mitsuba 0.5.0 for object insertion.

## High-Level structure
* data: data for object insertion.
* datasets: data reader functions.
* floormesh: floor mesh generation.
* layout_estimation: layout estimation.
* utils: helper functions.

## How to Run

### Step 1, Layout Estimation:
Please contact the author for the pre-trained model. 

```
cd layout_estimation
python run_inference_semantic.py
```

The output is in data/zind/scenes/layout_merge



### Step 2, Floor Mesh Generation:
```
cd floormesh
python floormesh.py
```

The output is in data/zind/scenes/floormesh


### Step 3, Object Insertion:
```
cd render
python object_insertion.py
```

It takes around 2 minutes to render a low resolution example. The output image is stored as render/render_obj.png


