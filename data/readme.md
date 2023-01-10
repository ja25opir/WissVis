<u>ridges_2d_uniform_cellbased.vtk</u>

1) grid:  

  * dimensions: 500 x 500 x 67107712 (?)
  * data: line 7 - 250.006  
    &mdash;> 250.000 gridpoints  
    &mdash;> 249.001 cells (499 x 499)

2) scalar:  

  * data: line 250.010 - 499.010

<u>ridges_2d_uniform_pointbased.vtk</u>

1) grid:  

  * dimensions: 500 x 500 x 67107712 (?)
  * data: line 7 - 250.006  
    &mdash;> 250.000 gridpoints  
    &mdash;> == 250.000 datapoints (500 x 500)

2) scalar:  

  * scalar data: line 250.010 - 500.009

<u>ridges_3d_unstructured_cellbased.vtk</u>

1) grid:  

  * dimensions: no information given  
    (looks like 500 x 500 x 500)
  * data: line 6 - 250.005  
    &mdash;> 250.000 gridpoints

2) cells:

  * dimensions: 249.001 (500 x 500), 1.245.005 (?)
  * info data: line 250.008 - 499.008
  * info format: dimension idx1 idx2 idx3 idx4 (idx = gridpoint index)
 
3) cell types:

 * line 499.010 - 748.010 (9 = quad)

4) scalar:

  * (cell) data: line 748.014 - 997.014
  * &mdash;> 249.001 (== cells)

<u>ridges_3d_unstructured_pointbased.vtk</u>

1) grid:  

  * dimensions: no information given  
    (looks like 500 x 500 x 500)
  * data: line 6 - 250.005  
    &mdash;> 250.000 gridpoints

2) cells:

  * dimensions: 249.001 (500 x 500), 1.245.005 (?)
  * info data: line 250.008 - 499.008
  * info format: dimension idx1 idx2 idx3 idx4 (idx = gridpoint index)

3) types:

  * line 499.010 - 748.010 (9 = quad)

3) scalar:

  * (point) data: line 748.014 - 998.013  
  &mdash;> 250.000 (== gridpoints)