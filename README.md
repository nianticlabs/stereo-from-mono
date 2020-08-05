# [Learning Stereo from Single Images](https://arxiv.org/abs/2008.01484)

**[Jamie Watson](https://scholar.google.com/citations?view_op=list_works&hl=en&user=5pC7fw8AAAAJ), [Oisin Mac Aodha](https://homepages.inf.ed.ac.uk/omacaod/), [Daniyar Turmukhambetov](http://dantkz.github.io/about), [Gabriel J. Brostow](http://www0.cs.ucl.ac.uk/staff/g.brostow/) and [Michael Firman](http://www.michaelfirman.co.uk) – ECCV 2020 (Oral presentation)**


[Link to paper](https://arxiv.org/abs/2008.01484)  
[Supplementary pdf](assets/stereo-from-mono-supplementary.pdf)  
[Video: ECCV 2 min presentation](https://storage.googleapis.com/niantic-lon-static/research/stereo-from-mono/short-video.mp4)  
[Video: ECCV 10 min presentation](https://storage.googleapis.com/niantic-lon-static/research/stereo-from-mono/long-video.mp4)


**Code is coming soon...**


<p align="center">
  <img src="assets/teaser.png" alt="Training data and results qualitative comparison" width="700" />
</p>

Supervised deep networks are among the best methods for finding correspondences in stereo image pairs. Like all supervised approaches, these networks require ground truth data during training. However, collecting large quantities of accurate dense correspondence data is very challenging. We propose that it is unnecessary to have such a high reliance on ground truth depths or even corresponding stereo pairs.

<p align="center">
  <img src="assets/method.png" alt="Overview of our stereo data generation approach" width="700" />
</p>

Inspired by recent progress in monocular depth estimation, we generate plausible disparity maps from single images. In turn, we use those flawed disparity maps in a carefully designed pipeline to generate stereo training pairs. Training in this manner makes it possible to convert any collection of single RGB images into stereo training data. This results in a significant reduction in human effort, with no need to collect real depths or to hand-design synthetic data. We can consequently train a stereo matching network from scratch on datasets like COCO, which were previously hard to exploit for stereo. 

<p align="center">
  <img src="assets/results.png" alt="Depth maps produced by stereo networks trained with Sceneflow and our method" width="700" />
</p>

Through extensive experiments we show that our approach outperforms stereo networks trained with standard synthetic datasets, when evaluated on  KITTI, ETH3D, and Middlebury. 

<p align="center">
  <img src="assets/table.png" alt="Quantitative comparison of stereo networks trained with Sceneflow and our method" width="600" />
</p>

## ✏️ 📄 Citation

If you find our work useful or interesting, please consider citing [our paper](https://arxiv.org/abs/2008.01484):

```
@inproceedings{watson-2020-stereo-from-mono,
 title   = {Learning Stereo from Single Images},
 author  = {Jamie Watson and
            Oisin Mac Aodha and
            Daniyar Turmukhambetov and
            Gabriel J. Brostow and
            Michael Firman
           },
 booktitle = {European Conference on Computer Vision ({ECCV})},
 year = {2020}
}
```


# 👩‍⚖️ License
Copyright © Niantic, Inc. 2020. Patent Pending. All rights reserved. Please see the license file for terms.
