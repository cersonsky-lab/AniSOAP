from typing import Any

class CGRCacheList:
    """
    This is a simple class that only exists to be used as a "private" cache
    for computed ClebschGordanReal matrices (self._cg)
    It only stores last n CG matrices computed for distinct l_max values.
    It will discard the oldest data if you try to store more distinct elements
    than initially given size, sort of like a fixed-size queue.
    """
    def __init__(self, size: int):
        """
        A constructor that makes an empty cyclic list.
        """
        self._size = size
        self.clear_cache()

    def keys(self) -> list:
        return self._keys

    def insert(self, key, value) -> None:
        if key not in self.keys():
            # Store (key, value) pair in cyclic list
            self._cyclic_list[self._next_ins_index] = (key, value)
            
            # Update list of keys currently in the list
            if len(self._keys) < self._size:
                self._keys.append(key)
            else:
                self._keys[self._next_ins_index] = key

            # Update the index at which the next element should be inserted.
            self._next_ins_index = (self._next_ins_index + 1) % self._size
            
    def get_val(self, key) -> Any:
        for element in self._cyclic_list:
            if element is not None and key == element[0]:
                return element[1]
        raise IndexError(f"The specified key {key} is not in the list. Current keys in the list are: {self._keys}")
    
    def clear_cache(self) -> None:
        self._next_ins_index = 0
        self._keys = []
        self._cyclic_list = [None] * self._size  # will be a list of tuples (key, value)