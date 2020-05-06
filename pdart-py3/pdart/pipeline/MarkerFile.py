import abc
from collections import namedtuple
import os
import os.path
import re
from typing import Optional


class MarkerInfo(object):
    def __init__(self, phase: str, state: str, text: Optional[str]):
        self.phase = phase.upper()
        self.state = state.upper()
        self.text = text

    def __str__(self) -> str:
        return f"MarkerInfo({self.phase!r}, {self.state!r}, {self.text!r})"


############################################################


class MarkerFile(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def clear_marker(self) -> None:
        pass

    @abc.abstractmethod
    def get_marker(self) -> Optional[MarkerInfo]:
        pass

    @abc.abstractmethod
    def set_marker(self, marker_info: MarkerInfo) -> None:
        pass


_RE = r"^#([A-Z_]+)#([A-Z_]+)#\.txt$"


def _info_to_path(info: MarkerInfo) -> str:
    return f"#{info.phase}#{info.state}#.txt"


class BasicMarkerFile(MarkerFile):
    def __init__(self, directory: str) -> None:
        self._directory = directory

    def _path_to_info(self, path: str) -> Optional[MarkerInfo]:
        m = re.match(_RE, os.path.basename(path))
        if m:
            with open(os.path.join(self._directory, path), "r") as f:
                text: str = f.read()
                # canonicalize empty contents to None
                if len(text) == 0:
                    return MarkerInfo(m.group(1), m.group(2), None)
                else:
                    return MarkerInfo(m.group(1), m.group(2), text)
        else:
            return None

    def get_marker(self) -> Optional[MarkerInfo]:
        for path in os.listdir(self._directory):
            res = self._path_to_info(path)
            if res:
                return res
        return None

    def clear_marker(self) -> None:
        old_info = self.get_marker()
        if old_info:
            old_path: str = os.path.join(self._directory, _info_to_path(old_info))
            os.remove(old_path)

    def set_marker(self, new_info: MarkerInfo) -> None:
        self.clear_marker()
        with open(os.path.join(self._directory, _info_to_path(new_info)), "w") as f:
            if new_info.text:
                f.write(new_info.text)
