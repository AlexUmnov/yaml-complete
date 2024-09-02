import yaml

from data_gather.filtering.base import BaseFilter

class NonEmptyFilter(BaseFilter):
    def __call__(self, text: str) -> bool:
        if text:
            return True
        else:
            return False

class ValidYamlFilter(BaseFilter):
    def __call__(self, text: str) -> bool:
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as e:
            return False

        if isinstance(loaded, dict) or isinstance(loaded, list):
            return True
        else:
            False

