mkdir yelp2018
python3 - << 'PY'
import gdown

url = "https://drive.google.com/drive/folders/1DKJdRRfRxNvvzbeF5k3kInHZNV1Org5N"
output = "./yelp2018" 

gdown.download_folder(url, output=output, quiet=False)
PY