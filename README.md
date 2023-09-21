# Ray-Patch: An Efficient Querying for Light Field Transformers

Official implementation of the paper ["Ray-Patch: An Efficient Querying for Light Field Transformers"](https://arxiv.org/abs/2305.09566).

<img src="https://drive.google.com/uc?export=view&id=11Ol27ifHZihLYiM157XpwCZD_zYIGOMM" alt="Querying comparison" width="512"/>
<img src="https://drive.google.com/uc?export=view&id=1-0clAkYwOCGMF0BOM71ij9Tv0NUJg6Q7" alt="Architecture" width="512"/>

## Results
### MSN-Easy 
* $60\times 80$

| Run | PSNR | SSIM | LPIPS | Rendering Speed | 
|---|---|---|---|---|
|`srt` |30.98 | 0.903 | 0.173 | 117 fps |
|`RP-srt k=2` |31.16| 0.906 | 0.163 | 288 fps |
|`RP-srt k=4` |30.92| 0.901 | 0.175 | 341 fps |

* $120\times160$

| Run | PSNR | SSIM | LPIPS | Rendering Speed | Checkpoint |
|---|---|---|---|---|---|
|`srt` |32.842| 0.935 | 0.250 | 192 fps |[Link](https://drive.google.com/file/d/1nJ7506fFm_xC1wKwyulIQ-pm4CkWnYx8/view)|
|`RP-srt k=4` |32.818| 0.935 | 0.254 | 275 fps |[Link](https://drive.google.com/file/d/1kz9yj8d_4WCFDGw3eJjE1nHYPcscJ0bg/view)|
|`RP-srt k=8` |32.306| 0.929 | 0.274 | 305 fps |[Link](https://drive.google.com/file/d/12pgck8Ymn7B6LTxV844cJwZz93g1WUbJ/view)|
|`osrt` |30.95| 0.916 | 0.287 | 21 fps |[Link](https://drive.google.com/file/d/1drZ4cPrxxsNREO78OrB4K_uIG4B8RTiP/view)|
|`RP-osrt k=8` |31.03| 0.915 | 0.303 | 278 fps |[Link](https://drive.google.com/file/d/1yr3tuGkK1fn1iT7-ShWhqx2c3YejxW0B/view)|


### ScanNet 
* In $240\times320$/Out $480\times640$

|Run | PSNR | SSIM | LPIPS | RMSE | Abs.Rel. | Square Rel.| Rendering Speed | Download |
|---|---|---|---|---|---|---|---|---|
|`DeFiNe`| 23.46 | 0.783 | 0.495 | 0.275 | 0.108 | 0.053 | 7 fps |[Link](https://drive.google.com/file/d/1C_RYqYXeNJjsO26ihZk3GZhMVxrNg4jQ/view)|
|`RP-DeFiNe k=16`| 24.54 | 0.801 | 0.453 | 0.263 | 0.103 | 0.050 | 208 fps |[Link](https://drive.google.com/file/d/1UtyNxAwj1B6kGWS8MpqtfYdUUsnu2pjS/view)|


## Setup
The implementation has been done using Pytorch 2, Pytorch Lightning 2, and cuda 11.7.
To run the repository we suggest to use the conda environment:

 * Clone the repository
    ``` 
    git clone git@github.com:tberriel/RayPatchQuerying.git
    ```
 * Create a conda environment
    ``` 
    conda env create -n PT2 --file=pt2.yml 
    conda activate PT2 
    ```

### Data
The models are evaluated on two datasets:
* MultiShapeNet-Easy dataset, introduced by [Stelzner et al.](https://stelzner.github.io/obsurf/): Download from [Link](https://drive.google.com/file/d/1RlHIbJ9NDtFgDBs1v0oQNhirXJak04UV)
* ScanNet dataset [Dai et al.](https://github.com/ScanNet/ScanNet): Follow the original repository instructions to acces the dataset. Then, to decode [NASDE](https://github.com/udaykusupati/Normal-Assisted-Stereo) stereo pairs used for training and evaluation by DeFiNe, follow these intructions:
    * After downloading ScanNet data, uncompress it with our modified scripts:
      ```
        cp /<path to RayPatch>/src/SensReader/* /<path to scannet>/ScanNet/SensReader/.
        cd /<path to scannet>/ScanNet/SensReader
        python decode.py --dataset_path /<path to scannet>/scans --output_path /<path to scannet>/data/val/ --split_file scannetv2_val.txt --frames_list frames_val.txt
        python decode.py --dataset_path /<path to scannet>/scans --output_path /<path to scannet>/data/train/ --split_file scannetv2_train.txt --frames_list frames_train.txt
      ```
    * Then run the following script to preprocess it:
      ```
        cd /<path to RayPatch>/
        python src/data/preproces_scannet.py /<path to scannet>/data/ /<path to RayPatch>/data/scannet/ --parallel --num-cores 12
        mv /<path to RayPatch>/data/stereo_pairs_* /<path to RayPatch>/data/scannet/.
      ```
      Preprocessing consist of resizing RGB data to 480x640 resolution. Set ``` --num-cores ``` to the number of cores of your cpu to process multiple scenes in parallel.
      
Ensure the data is placed in their respective folders:
  ```
  |-- RayPatch
     |-- data
       |-- msn_easy
          |-- train
          |-- val
          |-- test
       |-- scannet
          |-- train
          |-- val
  ```
## Experiments
Each training run should be stored inside the runs folder of the respective dataset, with its corresponding configuration file:
  ```
  |-- RayPatch
    |-- runs
        |-- scannet
          |-- define_32_stereo_acc
              |-- config.yaml
              |-- model_best.ckpt
          |-- rpdefine_16_32_stereo_acc
              |-- config.yaml
              |-- model_best.ckpt
  ```
### Test
To evaluate a model run:
```
  python test.py /<path to config file>/ --full-scale --eval-split <split>
```
To evaluate on MSN-Easy, use `--eval-split test`. For ScanNet use `--eval-split val`.

Add flag `--vis` to render a batch of images. Use flag `--num_batches` to set the number of batches to save.

By defalut evaluation does not compute neither LPIPS nor SSIM. To compute them add respective flags: 
```
  python test.py /<path to config file>/ --full-scale --lpips --ssim
```
SSIM computation has a huge memory footprint. To evaluate `define_stereo_32` we had to run evaluation on CPU with 160 GB of RAM.

To evaluate profiling metrics for render a single image run:
```
python test_profile.py <path to config folder> --batch --flops --time
```
The `<path to config folder>` should be like `runs/scannet/define_32_stereo_acc/`.

To execute on a GPU device add `--cuda` flag.

### Train
To train a model run:
```
  python train.py /<path to config file>/ 
```
To train a Ray-Patch querying model add argument `--full-scale`.

Training the model also has a huge memory footprint. We trained both models using 4 Nvidia Tesla V100 with 32 GBytes of VRAM each. For ScanNet experiments, we used batch 16 and gradient accumulation of 2 to simulate batch size 32.
