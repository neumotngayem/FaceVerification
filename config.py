import yaml

class Config():
    def __init__(self, config_path='config.yml'):
        self.config_path = config_path
            
    def parse_config(self):
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)