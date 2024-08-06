# Problem Description

### Classify CT scans into :

#### 0: notumor
- Definition: This indicates the absence of a tumor. Brain images classified as "notumor" are normal and show no signs of abnormal growths.
- Characteristics: No presence of mass or lesion. The brain's structure appears normal in imaging scans.
- Training Example Path: `Tr-no_0010.jpg`
- Testing Example Path: `Te-no_0010.jpg`

#### 1: pituitary
- Definition: Pituitary tumors are growths that occur in the pituitary gland, a small gland located at the base of the brain.
- Characteristics:
Often benign (non-cancerous).
Can affect hormone production, leading to various endocrine disorders.
Symptoms may include vision problems, headaches, and hormonal imbalances (e.g., Cushing's disease, acromegaly).
- Training Example Path: `Tr-pi_0010.jpg`
- Testing Example Path: `Te-pi_0010.jpg`

#### 2: meningioma
- Definition: Meningiomas are tumors that arise from the meninges, the membranes that surround the brain and spinal cord.
- Characteristics:
Usually benign but can be malignant in rare cases.
Slow-growing and can become quite large before causing symptoms.
Symptoms depend on the tumor's location and size and can include headaches, seizures, and neurological deficits.
- Training Example Path: `Tr-me_0010.jpg`
- Testing Example Path: `Te-me_0010.jpg`

#### 3: glioma
- Definition: Gliomas are a type of tumor that originates in the glial cells, which are supportive cells in the brain.
- Characteristics:
Can be benign (low-grade) or malignant (high-grade).
Common types include astrocytomas, oligodendrogliomas, and ependymomas.
Symptoms can vary but often include headaches, seizures, and neurological deficits depending on the tumor's location.
- Training Example Path: `Tr-gl_0010.jpg`
- Testing Example Path: `Te-gl_0010.jpg`

# Data
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data

```
unzip [Archive.zip] -d data
```

# Environment
```
> conda create --name vit-env python=3.11
> source activate vit-env
> pip install -r requirements.txt
> pip install -r requirements-dev.txt
```