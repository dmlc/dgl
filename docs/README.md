DGL document and tutorial folder
================================


To build the doc:

- Create the developer conda environment using the script [here](../script/create_dev_conda_env.sh).
- Activate the developer conda environment.
- Build the doc using the script [here](../script/build_doc.sh).

To render locally:
```
cd build/html
python3 -m http.server 8000
```
