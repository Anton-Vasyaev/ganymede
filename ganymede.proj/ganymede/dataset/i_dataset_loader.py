# python
from abc    import abstractmethod
from typing import Generic, TypeVar, Tuple

InputT  = TypeVar('InputT')
TargetT = TypeVar('TargetT')


class IDatasetLoader(Generic[InputT, TargetT]):
    @abstractmethod
    def __len__(self) -> int:
        '''
        Returns numbers of examples.

        Returns:
            int: numbers of examples.
        '''
        raise NotImplementedError()
    

    @abstractmethod
    def __getitem__(self, idx : int) -> Tuple[InputT, TargetT]:
        '''
        Return example of input and target.

        Args:
            idx (int): Index of example

        Returns:
            Tuple[InputT, TargetT]: Input and target.
        '''
        raise NotImplementedError()
