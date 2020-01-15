# Dissertation Project - Master of Science (MSc)
 	         
## Title
	Modelling Tongue Movements

## Description
 “Modelling Tongue Movements” is a computer vision domain related project that aims to address the problem of locating the tongue in ultrasound tongue images (UTI). In order to achieve this, a statistical shape model (SSM) is built primarily. The obtained mean shape model is then used as a reference to identify and locate the exteriors of a tongue in UTIs. Active shape model (ASM) and random forest regression voting (RFRV) methods are used to carry out the process of searching (locating) tongue in ultrasound tongue images. The observed error rates of the respective methods and their overall performance provide a means to compare and evaluate the two methods. After evaluation, it is observed that the random forest regression voting method performs better compared to the active shape model due to various factors. The methods, their results and the factors affecting them are addressed in my thesis.

## Languages/Tools used
 ASM annotation tool, Python, OpenCV, Constrained local model optimiser, BoneFinder.

## Outcome
   Modelling tongue movements a computer vision domain related project aims at identifying and locating tongue contours in ultrasound tongue images (UTI). This is achieved in order to further investigate and address the issue of tongue lateralization. This thesis studies statistical shape models for building tongue shape models. The data was annotated and was used for building tongue shape models. During this process, an essential research question of how to design and model tongue was addressed.

The mean shape models obtained after training and aligning the dataset were investigated for deciding on the constraints of deformability. A number of observations were made during this process. Factors affecting variations were realized. Methods were derived to restrict the variation to a degree so as to preserve the meaning of the modelled shape. In this case, to preserve the meaning of the modelled tongue. In this process, another crucial research question on capturing the variation of the modelled shape was discussed.

This thesis proceeds further in investigating the effectiveness of two image search approaches. Namely, active shape models (ASM) and random forest regression voting (RFRV) methods. The literature of these two methods was studied and implemented. Several diverse challenges were encountered. These challenges are acknowledged and addressed in this thesis. A test and evaluation plan were drafted to evaluate the effectiveness of the two methods. 

The models using both the methods were trained and tested using 10-fold cross validation. The root mean squared error (RMSE) of the two methods are compared. With sufficient proof, it was concluded that the random forest regression voting method is a more efficient, adequate, optimal and robust method compared to the active shape modelling method.
