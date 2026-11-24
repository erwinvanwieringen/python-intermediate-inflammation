import glob
import json
import os
from abc import ABC, abstractmethod

from inflammation import models


class DataSource(ABC):
    pass

    @abstractmethod
    def load_inflammation_data(self):
        pass


class CSVDataSource(DataSource):
    def __init__(self, path, pattern):
        self._path = path
        self._pattern = pattern

    def load_inflammation_data(self):
        data_files = glob.glob(os.path.join(self._path, self._pattern))
        if len(data_files) == 0:
            raise ValueError(
                f"No inflammation data CSV files found in path {self._path}"
            )
        return list(map(models.load_csv, data_files))
    
class JSONDataSource(DataSource):
    def __init__(self, path, pattern):
        self._path = path
        self._pattern = pattern
        
    def load_inflammation_data(self):
        data_files = glob.glob(os.path.join(self._path, self._pattern))
        if len(data_files) == 0:
            raise ValueError(
                f"No inflammation data JSON files found in path {self._path}"
            )
        return list(map(models.load_json, data_files)) 
