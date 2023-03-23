# python
from abc    import abstractmethod
from typing import Generic, TypeVar, Tuple

InputT  = TypeVar('InputT')
TargetT = TypeVar('TargetT')


class IDatasetLoader(Generic[InputT, TargetT]):
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()
    

    @abstractmethod
    def __getitem__(self, idx : int) -> Tuple[InputT, TargetT]:
        raise NotImplementedError()
