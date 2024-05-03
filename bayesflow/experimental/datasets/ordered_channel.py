
import ctypes
import multiprocessing as mp


class OrderedChannel:
    """
    Implements an ordered Multi Producer, Single Consumer (MPSC) Channel
    As with a regular MPSC channel, there is a single receiver and many senders.
    Unlike a regular MPSC channel, items are yielded not in the order they are sent,
    but by a sequential index. This makes it similar to a priority queue, with the
    difference that we always wait for the item with the next highest priority.

    The primary use case for this class is multiprocess loading of ordered datasets.

    Examples:
        >>> oc = OrderedChannel(10)
        >>> oc.put(0, "zero")
        >>> zero = oc.get()
        >>> assert zero == "zero"

        >>> oc = OrderedChannel(10)
        >>> oc.put(1, "one")
        >>> oc.put(0, "zero")
        >>> zero = oc.get()
        >>> assert zero == "zero"
        >>> one = oc.get()
        >>> assert one == "one"

        >>> oc = OrderedChannel(10)
        >>> oc.get()  # blocks until index 0 is filled

        >>> oc = OrderedChannel(1)
        >>> oc.put(0, "zero")
        >>> oc.put(1, "one")  # blocks until index 0 is emptied
    """
    def __init__(self, maxsize: int):
        self.lock = mp.RLock()

        # TODO: use a heap instead?
        self.manager = mp.Manager()
        self.items = self.manager.dict()
        self.index = self.manager.Value(ctypes.c_uint64, 0)
        self.maxsize = self.manager.Value(ctypes.c_uint64, maxsize)

        # TODO: use an RWLock with writer priority
        self.put_condition = mp.Condition(self.lock)
        self.get_condition = mp.Condition(self.lock)

    def put(self, index: int, item: any) -> None:
        # block until we can read/write safely
        with self.lock:
            if index < self.index.value or index in self.items:
                raise ValueError(f"Index {index} was already seen.")

            # block if the index is out of bounds wrt maxsize
            self.put_condition.wait_for(lambda: index < self.index.value + self.maxsize.value)

            # block until there is space for the item in the heap
            # the lock is released while we wait and reacquired after
            self.put_condition.wait_for(lambda: len(self.items) < self.maxsize.value)

            # can safely push now
            self.items[index] = item

            # notify other threads that we just added an item
            self.get_condition.notify()

    def get(self) -> any:
        # block until we can read/write safely
        with self.lock:
            # block until the item is in the heap
            # the lock is released while we wait and reacquired after
            self.get_condition.wait_for(lambda: self.index.value in self.items)

            # can safely pop now
            item = self.items.pop(self.index.value)
            self.index.value += 1

            # notify other threads that we just removed an item
            self.put_condition.notify()

            return item
