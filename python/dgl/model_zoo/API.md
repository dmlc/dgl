Model Zoo API
==================

We tend to provide two major API for model zoo. We will focus on PyTorch for the first release. 

- `model_zoo.chem.[Model_Name]` to load the model skeleton
- `model_zoo.chem.load_pretrained([Pretrained_Model_Name])` to load the model with pretrained weights

Models would be placed in `python/dgl/model_zoo/chem`.

Each Model should contains the following elements:
- Papers related to the model
- Model's input and output
- What dataset could use this
- Documentation for all the customizable configs
- Credits (Contributor infomation)

