# Long-Term Typhoon Trajectory Prediction: A Physics-Conditioned Approach Without Reanalysis Data

This is the official project repository for [Long-Term Typhoon Trajectory Prediction: A Physics-Conditioned Approach Without Reanalysis Data](https://openreview.net/forum?id=ziDFH8TPPK) (ICLR 2024 Spotlight).

## Abstract
In the face of escalating climate changes, typhoon intensities and their ensuing damage have surged. Accurate trajectory prediction is crucial for effective damage control. Traditional physics-based models, while comprehensive, are computationally intensive and rely heavily on the expertise of forecasters. Contemporary data-driven methods often rely on reanalysis data, which can be considered to be the closest to the true representation of weather conditions. However, reanalysis data is not produced in real-time and requires time for adjustment because prediction models are calibrated with observational data. This reanalysis data, such as ERA5, falls short in challenging real-world situations. Optimal preparedness necessitates predictions at least 72 hours in advance, beyond the capabilities of standard physics models. In response to these constraints, we present an approach that harnesses real-time Unified Model (UM) data, sidestepping the limitations of reanalysis data. Our model provides predictions at 6-hour intervals for up to 72 hours in advance and outperforms both state-of-the-art data-driven methods and numerical weather prediction models. In line with our efforts to mitigate adversities inflicted by typhoons, we release our preprocessed PHYSICS TRACK dataset, which includes ERA5 reanalysis data, typhoon best-track, and UM forecast data.


## Poster
![ICLR_Poster](https://github.com/iclr2024submit/LT3P/assets/146421749/52d9e6d8-5dc6-41cf-a17b-53f1c6395785)

## LT3P Inference code

**UM data link**: [https://drive.google.com/drive/folders/1_hos41VlpbFIlhjCcuniN8OTCKe4ctBu?usp=drive_link]

**Best Track link**: [https://drive.google.com/drive/folders/1fyiXUdseyUVCabqT7JcxxblTpXup1RVR?usp=drive_link]

**Additional Visualization Results** [https://drive.google.com/drive/folders/1-ZTpPgQ0YRdxL0xzdTvC0fYz6hKLKLdf?usp=drive_link]

**pre-trained weight** [https://drive.google.com/drive/folders/1KGuyhXbKbFZCsJQ1x6EhyrQQWhIqIt27?usp=drive_link]



Note that because the capacity of UM data is very large, the size that can be uploaded anonymously is limited.  We only disclose test data.

Folder Structure
```
./README.md
./infer.py
./lt_tpc.py
./metrics.py
./model.py
./modules.py
./utils.py
./UM_2019/
---------/.npy
./2019/
---------/.txt
./1900.pth

```

```bash
python3 infer.py
```

### Visualization results are automatically saved. 

The red dot is prediction, the green dot is input, and the blue dot is GT.
![bwp292019_KAMMURI_part1](https://github.com/iclr2024submit/LT3P/assets/146421749/2707dc71-27fc-4df6-b263-ce8165f0bd8d)
![bwp292019_KAMMURI_part3](https://github.com/iclr2024submit/LT3P/assets/146421749/4f0da225-47eb-4a3b-b4d0-5348284ddc7a)


Note that we provide a deterministic model due to inference code compatibility.


***

## Citation
```bibtex
@inproceedings{park2023long,
  title={Long-Term Typhoon Trajectory Prediction: A Physics-Conditioned Approach Without Reanalysis Data},
  author={Park, Young-Jae and Seo, Minseok and Kim, Doyi and Kim, Hyeri and Choi, Sanghoon and Choi, Beomkyu and Ryu, Jeongwon and Son, Sohee and Jeon, Hae-Gon and Choi, Yeji},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
