# Introduction
This repository demonstrates application of Fourier Serie on drawing any 2D line-art painting
# Requirement
```
opencv-python                 4.5.5.62
numpy                         1.22.3
svg.path                      6.2
```
# How to run
Step 1: Download a line-art image from [Pinterest](https://www.pinterest.com/search/pins/?q=1%20line%20art&rs=typed).  
Step 2: Use [Adobe tool](https://www.adobe.com/express/feature/image/convert/svg) to convert the image to svg format.  
Step 3: Run

```bash
python main.py --svg svg/cat_2.svg --ppp 0.5
```
![cat](gif/cat_2.gif)
