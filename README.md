# Gaussian Process Hierarchy
A modification of Gaussian Process Latent Variable Models for data integration taking into account account the hierarchical nature of the data. Work initiated during a Postdoc in CREATIS by Gabriel Bernardino, and completed during Benoit Freiche's PhD. 

The rationale is to exploit data that contains different levels of resolution / granuality, similarly to a multi-scale approach.

## Application: Cardiac shapes

Gabriel:

We use a dataset of End-Diastolic biventricular cardiac shapes, obtained through MRI using a deformable template. We identified two levels:
- A simple description, based on 3 global scalar measurements that are commonly used in clinical practice: right ventricular ED volume (RV-EDV),  left ventricular ED volume (RV-EDV) and left ventricular myocardial mass (LVM).
- The shape itself, encoded via a Point Distribution Model


We used a dataset of MR images segmentations from MIMI cohort [1]
## Subpackages:
- models : 
- base :
- readingUtilsMRI : 
## Dependencies


## Usage

Installation with python 3.11 (BF 18/04/24):
in anaconda prompt:

cd .../Code_these_pour_ND/gpHierarchy
pip install -e .
pip install tensorflow
pip install gpflow
pip install tf_keras
pip install GPy
pip install pyvista

There shouldn't be problems with dependencies now

## References

TODO