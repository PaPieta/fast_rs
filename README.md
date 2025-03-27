# Fast sphericity and roundness approximation in 2D and 3D 

Fast approximation of Wadell sphericity and roundness in 2D and 3D, described in our paper [TODO:link]

The method uses the output from local thickness algorithm to approximate the two measures. Resulting sphericity values closely match those from exact approaches. Roundness values no longer retain the original 0-1 range, but correlate closely to it.

The execution speed is dependent primarily on image volume, and not on number of objects, enabling fast evaluation of thousands of objects at once.

**For in-depth introduction to the library, visit the [CodeOcean capsule](https://codeocean.com/capsule/9246661/tree/v1).**

## Installation

``` python
pip install git+https://github.com/PaPieta/fast_rs.git
```

## Prerequisites

Library requires ```numpy```, ```scipy```, ```scikit-image```, ```localthickness```. They can be installed via provided requirements file:

```sh
  pip install -r requirements.txt
```

## How to use

>For in-depth introduction, visit the [CodeOcean capsule](https://codeocean.com/capsule/9246661/tree/v1).

Both sphericity and roundness can be calculated together, from a binary mask. The process is exactly the same in 2D and 3D.

``` python
from fast_rs import rs

mask = ... # Load/provide binary mask (can be both a 2D and 3D np.array)
roundness_vec, sphericity_vec, label_img = rs.rs_calulate(mask)
```

Returned label image indices can be used to connect measure values to specific objects in the mask.

**Example result:**

<img src="doc_img\demo_img.png" alt="drawing" width="500"/>

## Paper

The fast sphericity and roundness approximation method is described and evaluated in our contribution to (TBA). Please cite our paper if you use the method in your work.

```
TBA
```

## License

MIT License (see LICENSE file).