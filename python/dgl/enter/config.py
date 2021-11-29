import yaml

with open('example.yaml') as f:
    data = yaml.load(f.read())
print(data)
