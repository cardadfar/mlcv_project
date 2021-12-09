# MLCV Project


## Setup

Download a subset of the [data](https://drive.google.com/file/d/1kfZsjTI27OfSGDJF89R9o_3WErdboXnM/view?usp=sharing) and place them in data

Download the [checkpoints](https://drive.google.com/drive/folders/11fBNOeUny9wDw0SHeLEGWjiV7yPZFPQP?usp=sharing) and place them in checkpoints

## Testing

Place any images of class 'classname' in 'data/png/classname'. For example, interpolating two images of broccoli should be placed in 'data/png/broccoli'. Images should be 256 x 256 and grayscale.

Make sure to set the batch size equal to the number of images. For example, if interpolating between 5 images, run:

```
python3 test.py --batch-size 5 --img-class broccoli
```

When running interpolation, outputs will be saved to 'results/test'.

## Changing Interpolation

On lines 92-93 in 'test.py'

```                
y = linear(encoded, ibf=20, ease=sigmoid)
y = catmullRom(encoded, ibf=20, ease=sigmoid)
```

You can configure the spatial interpolation (linear/catmulRom), the number of inbetween frames (ibf) and the easing amount (linear/sigmoid). For the linear call, you may also pass in a set of interpolation points p. For example, 

```                
y = linear(encoded, ibf=20, ease=sigmoid, p=[(0.2,-1), (0.8,2), (1,1)])
```

Will run linear spatial interpolation with sigmoid easing with anticipation and follow-through.
