import yaml

info = []

with open(r'\config.yaml', 'w') as file:
    documents = yaml.dump(info, file)