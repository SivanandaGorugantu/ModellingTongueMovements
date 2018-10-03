# ModellingTongueMovements
Modelling Tongue Movements – a computer vision domain related project, locating tongue contour in ultrasound tongue images (UTI) based on active shape models and random forest regression voting approaches.
This project encloses only the active shape modelling (ASM) model files. There are 4 models. 
Namely, the universal model (ASM.py), model representing sound a (ASM_a.py), model representing sound o (ASM_o.py) and model representing sound l (ASM_l.py). 
Each model implements 10-fold cross-validation. 
Keeping processing time in mind, the train and test folds (or their range) can be set manually (can be modified in the code).
