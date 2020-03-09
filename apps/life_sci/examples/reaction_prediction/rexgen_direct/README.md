# A graph-convolutional neural network model for the prediction of chemical reactivity

- [paper in Chemical Science](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc04228d#!divAbstract)
- [authors' code](https://github.com/connorcoley/rexgen_direct)

An earlier version of the work was published in NeurIPS 2017 as 
["Predicting Organic Reaction Outcomes with Weisfeiler-Lehman Network"](https://arxiv.org/abs/1709.04555) with some 
slight difference in modeling.

## Dataset

The example by default works with reactions from USPTO (United States Patent and Trademark) granted patents, 
collected by Lowe [1]. After removing duplicates and erroneous reactions, the authors obtain a set of 480K reactions. 
The dataset is divided into 400K, 40K, and 40K for training, validation and test.

## Reaction Center Prediction

Reaction centers refer to the pairs of atoms that lose/form a bond in the reactions. A Graph neural network 
(Weisfeiler-Lehman Network in this case) is trained to update the representations of all atoms. Then we combine 
the atom representations to predict the likelihood 

## References

[1] D. M.Lowe, Patent reaction extraction: downloads, 
https://bitbucket.org/dan2097/patent-reaction-extraction/downloads, 2014.
