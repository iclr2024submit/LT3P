# LT3P Inference code

**UM data link**: [https://drive.google.com/drive/folders/1_hos41VlpbFIlhjCcuniN8OTCKe4ctBu?usp=drive_link]

**Best Track link**: [https://drive.google.com/drive/folders/1fyiXUdseyUVCabqT7JcxxblTpXup1RVR?usp=drive_link]

**Additional Visualization Results** [https://drive.google.com/drive/folders/1-ZTpPgQ0YRdxL0xzdTvC0fYz6hKLKLdf?usp=drive_link]

**pre-trained weight** [https://drive.google.com/file/d/1ShXQ1lA6zKoyDvugY79NbczIWhz5Mjig/view?usp=drive_link]

# Poster
[ICLR_Poster.pdf](https://github.com/user-attachments/files/15574564/ICLR_Poster.pdf)


Note that because the capacity of UM data is very large, the size that can be uploaded anonymously is limited.  We only disclose test data.
Full data link will be released later.

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

Our full training and inference code will be released after review.
***
