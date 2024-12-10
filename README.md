# Hierarchical Gaussian Process Latent Variable Models
A modification of Gaussian Process Latent Variable Models for data integration taking into account account the hierarchical nature of the data. Work initiated during a Postdoc in CREATIS by Gabriel Bernardino, and completed during Benoit Freiche's PhD. 

The rationale is to exploit data that contains different levels of resolution / granuality, similarly to a multi-scale approach.

This code corresponds to the following paper:

B. Freiche, G. Bernardino, R. Deleat-Besson, P. Clarysse and N. Duchateau (2024) Hierarchical data integration with Gaussian processes: application to the characterization of cardiac ischemia-reperfusion patterns, IEEE Transactions on Medical Imaging, https://doi.org/10.1109/TMI.2024.3512175

## Application: CelebA
 The orginal focus of the paper [1] is the characterization of cardiac ischemia-reperfusion patterns, using an MR imaging dataset, the MIMI database [2]. In this repository, we rather illustrate the method on a public imaging dataset, a reduced version of CelebA. This version can be downloaded here #TODO.

## Installation

Installation with python 3.11:
in anaconda prompt:

pip install -e . 

pip install tensorflow 

pip install gpflow 

pip install tf_keras 

pip install GPy 

pip install pyvista

## References

[1] B. Freiche, G. Bernardino, R. Deleat-Besson, P. Clarysse and N. Duchateau (2024) Hierarchical data integration with Gaussian processes: application to the characterization of cardiac ischemia-reperfusion patterns, IEEE Transactions on Medical Imaging, https://doi.org/10.1109/TMI.2024.3512175

[2] Belle L et al. Comparison of Immediate With Delayed Stenting Using the Minimalist Immediate Mechanical Intervention Approach in Acute ST-Segment-Elevation Myocardial Infarction: The MIMI Study. Circ Cardiovasc Interv. 2016 Mar;9(3):e003388. doi: 10.1161/CIRCINTERVENTIONS.115.003388. PMID: 26957418.
