from xml.dom import NoDataAllowedErr
from interfaces import Model
from typing import Type, Dict, Union

class PlateDetection:
    """_summary_
    """
    
    def __init__(self, model: Type[Model]) -> None:
        self.__model = model
    
    def first_rule(self, data: bool) -> Union[Dict, None]:
        if data:
            return self.__model.create_model()
        raise NoDataAllowedErr