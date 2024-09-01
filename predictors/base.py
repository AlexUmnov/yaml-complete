from abc import ABCMeta, abstractmethod

class AutocompletePredictor(metaclass=ABCMeta):
    @abstractmethod
    def predict(
        self,
        text_before: str, 
        text_after: str
    ) -> str:
        # Get's current file's text in 3 parts, before, after and current line
        # Return completion for current line
        pass