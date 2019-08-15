Model Zoo API
==================

We provide two major APIs for the model zoo. For the time being, only PyTorch is supported.
- `model_zoo.chem.[Model_Name]` to load the model skeleton
- `model_zoo.chem.load_pretrained([Pretrained_Model_Name])` to load the model with pretrained weights

Models would be placed in `python/dgl/model_zoo/chem`.

Each Model should contain the following elements:
- Papers related to the model
- Model's input and output
- Dataset compatible with the model
- Documentation for all the customizable configs
- Credits (Contributor infomation)
