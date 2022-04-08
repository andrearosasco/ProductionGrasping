import abc


class ABCGetAttrMeta(abc.ABCMeta):
    def __getattribute__(self, item):
        print('Getting attribute', item)
        return abc.ABCMeta.__getattribute__(self, item)


class sysprops(object, metaclass=ABCGetAttrMeta):


    @property
    @abc.abstractmethod
    def run(self) -> str:
        """" A string pointing to the script to run"""
        pass

class test(sysprops):
    run = 'ciao'
    ciao = 'hey'

if __name__ == '__main__':
    test().ciao