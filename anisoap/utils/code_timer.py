import time
from collections import defaultdict
from typing import Callable
from enum import Enum


class SimpleTimerCollectMode(Enum):
    AVG = 1,
    SUM = 2,
    MIN = 3,
    MAX = 4,
    MED = 5,

class SimpleTimer:
    # NOTE: Change this collect_mode default argument to change all measuring across the files
    #       except the one specified.
    default_coll_mode = SimpleTimerCollectMode.AVG

    def __init__(self):
        self._internal_time = time.perf_counter()
        self._timer_dict = defaultdict(list)

    def mark_start(self):
        """
        Sets the internal time to current time. Acts like a "reset" on stopwatch timer
        """
        self._internal_time = time.perf_counter()
        

    def mark(self, label: str) -> float:
        """
        Compute elapsed time from last "mark" or "mark_start" call and stores the
        duration to an internal dictionary, with key given by "label."
        It will append to list if label already exists, and it will create a new
        list if label does not exist.
        """
        curr_time = time.perf_counter()
        duration = curr_time - self._internal_time
        self._timer_dict[label].append(duration)
        self._internal_time = curr_time
        return duration
    
    def sorted_dict(self):
        """
        Returns a dictionary sorted by keys. However, the sort is special such that
        if the key contains leading integers, it will sort by integers first then
        sort by rest of the keys. Any key (label) with no leading integer will be
        treated as "infinity" so that it comes after others.
        """
        # get list of (largest leading number in key, key) tuples
        # calling "sorted" on list of tuples will sort the element by first item in
        # tuple then the second sort by the tuple
        keys = sorted([(self._largest_leading_num(k), k) for k in self._timer_dict])
        keys = [k for (_, k) in keys]    # discard the leading number part
        return { key: self._timer_dict[key] for key in keys }
    
    def collect_trials(self, collect_fn: Callable[[list[float]], float]) -> dict[str, float]:
        """
        Averages the list
        """
        coll_dict = defaultdict(float)
        for key, val in self._timer_dict.items():
            if len(val) == 0:
                coll_dict[key] = 0.0
            else:
                coll_dict[key] = collect_fn(val)
        return coll_dict
    
    def collect_and_append(
            self,
            other: 'SimpleTimer',
            collect_mode: SimpleTimerCollectMode | Callable[[list[float]], float] = None
        ):
        """
        Takes another SimpleTimer class as argument and calls average_trials
        in "other" then appends the averaged values into the internal list of
        "self" dictionary
        """
        # If None, use static variable "default_coll_mode" as collecting mode
        if collect_mode == None:
            collect_mode = SimpleTimer.default_coll_mode

        if collect_mode == SimpleTimerCollectMode.AVG:
            coll_dict = other.collect_trials(lambda x: sum(x) / len(x))
        elif collect_mode == SimpleTimerCollectMode.SUM:
            coll_dict = other.collect_trials(lambda x: sum(x))
        elif collect_mode == SimpleTimerCollectMode.MAX:
            coll_dict = other.collect_trials(lambda x: max(x))
        elif collect_mode == SimpleTimerCollectMode.MIN:
            coll_dict = other.collect_trials(lambda x: min(x))
        elif collect_mode == SimpleTimerCollectMode.MED:
            coll_dict = other.collect_trials(lambda x: SimpleTimer._median(x))
        else:
            coll_dict = other.collect_trials(lambda x: collect_mode(x))
        for key, val in coll_dict.items():
            self._timer_dict[key].append(val)

    def clear_time(self):
        """
        Clears all internal data of the timer and resets current time
        """
        self._timer_dict = defaultdict(list)
        self.mark_start()

    @staticmethod
    def _largest_leading_num(text: str) -> float:
        """
        This method returns float because it returns "inf" for things
        that do not have leading number. "floats" will always be some integer
        value.
        """
        int_str = ""
        for character in text:
            if character.isdigit():
                int_str += character
            else:
                break

        return float(int_str) if len(int_str) > 0 else float("inf")
    
    @staticmethod
    def _median(x: list[float]):
        sorted_x = sorted(x)
        len_x = len(x)
        half_index = len_x // 2 # floor

        if len_x % 2 == 0:  # even number of elements:
            return (sorted_x[half_index - 1] + sorted_x[half_index]) / 2.0
        else:
            return sorted_x[half_index]
