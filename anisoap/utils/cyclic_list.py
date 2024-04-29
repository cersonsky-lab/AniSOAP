from typing import List
class CGRCacheList:
    """
    This is a simple class that only exists to be used as a "private" cache
    for computed ClebschGordanReal matrices (self._cg)

    It only stores last n CG matrices computed for distinct l_max values. Since
    it is specialized to store (l_max, cg matrix) pair, it is NOT supposed to be
    used to cache any other things. While type of "value" (in `(key, value)` pair)
    does not matter as much, the type of `key` MUST BE A 32-BIT INTEGER with
    most significant bit unused, as it performs bitwise operations on the most
    significant bit to store and check the replacement flags.

    It will replace the entries using something called "clock algorithm" for page
    replacement, which mimics the "least recently used" algorithm but with less
    computation requirement.

    [Link to clock algorithm](https://en.wikipedia.org/wiki/Page_replacement_algorithm#Clock)
    """

    # Perform bitwise AND with this number to remove the replacement flag (set to 0).
    _REMOVE_RFLAG = 0x7FFFFFFF
    # Perform bitwise AND with this number to get replacement flag (0: replace, not 0: do not replace)
    # Or perform bitwise OR then assign the value to set the flag to 1.
    _GET_RFLAG = 0x80000000

    def __init__(self, size: int):
        """
        A constructor that makes an empty cyclic list.
        """
        self._size = size
        self.clear_cache()

    def keys(self) -> List[int]:
        """
        Return list of keys, with keys
        """
        return [
            entry[0] & CGRCacheList._REMOVE_RFLAG
            for entry in self._cyclic_list
            if entry is not None
        ]

    def insert(self, key: int, value: dict) -> None:
        """
        This insert algorithm mimics the "clock algorithm" for page replacement
        technique. The algorithm works like this:
        1. Get the entry that the "clock hand" (self._ins_index) points to.
        2. Keep advancing the "clock hand" until key with replacement flag = 0
        is found. Replacement flag, in this case, is the most significant bit
        3. Insert the new [key, value] pair into the list, with replacement flag = 1.
        Advance "clock hand" by one position.
        Note that clock hand wraps around the list, should it reach the end of the list.
        """
        if key not in self.keys():
            curr_entry = self._cyclic_list[self._ins_index]

            # Advance _ins_index until replacement flag is 0.
            while (
                curr_entry is not None and curr_entry[0] & CGRCacheList._GET_RFLAG != 0
            ):
                curr_entry[0] = curr_entry[0] & CGRCacheList._REMOVE_RFLAG
                self._ins_index = (self._ins_index + 1) % self._size
                curr_entry = self._cyclic_list[self._ins_index]

            # Store [key with replace_flag = 0, value] pair in cyclic list
            self._cyclic_list[self._ins_index] = [key, value]

            # Advance the "clock hand" once more after insertion.
            self._ins_index = (self._ins_index + 1) % self._size

    def get_val(self, key: int) -> dict:
        """
        Obtains the value for given key. Raises IndexError if the given key
        is not found in the list.
        """
        for element in self._cyclic_list:
            if element is not None and key == element[0] & CGRCacheList._REMOVE_RFLAG:
                element[0] = element[0] | CGRCacheList._GET_RFLAG
                return element[1]
        raise IndexError(
            f"The specified key {key} is not in the list. Current keys in the list are: {self.keys()}"
        )

    def clear_cache(self) -> None:
        """
        Resets the cache (makes list empty)
        """
        self._ins_index = 0
        # will be a list of list [key | replace_flag, value].
        self._cyclic_list: list = [None] * self._size
