To set-up your environment:
```
ssh g38nXX # XX:01-16
export LD_LIBRARY_PATH=/usr/local/cuda/lib
export PATH=$PATH:/usr/local/cuda/bin
```

For exercise 1:
```
nvcc dot_product.cu -o dot_product 
./dot_product
```

For exercise 2:
```
nvcc stencil_2d.cu -o stencil_2d 
./stencil_2d
```