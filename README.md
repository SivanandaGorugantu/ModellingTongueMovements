# ModellingTongueMovements
Modelling Tongue Movements – a computer vision domain related project, locating tongue contour in ultrasound tongue images (UTI) based on active shape models and random forest regression voting approaches.
This project encloses only the active shape modelling (ASM) model files. There are 4 models. 
Namely, the universal model (ASM.py), model representing sound a (ASM_a.py), model representing sound o (ASM_o.py) and model representing sound l (ASM_l.py). 
Each model implements 10-fold cross-validation. 
Keeping processing time in mind, the train and test folds (or their range) can be set manually (can be modified in the code).
The annotated files represent the landmark points. These files have .pts extention. Some examples can be found here. There are a total of 1177 such annotated files.
Some ultrasound tongue images can also be found here. There are a total of 2900 such data files, of which, 1177 were annotated and used for training the universal model. Ultrasound images and their respective annotated landmark point files representing individual sounds were used to train individual models representing sounds a, l and o.
