# python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class IModel(Generic[InputType, OutputType]):
    def forward(self, input : InputType) -> OutputType:
        raise NotImplementedError()