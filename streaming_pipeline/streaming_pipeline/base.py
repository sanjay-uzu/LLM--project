from threading import Lock

class SingletonMeta(type):
    """A thread-safe implementation of Singleton pattern using metaclass.
    
    This metaclass ensures that only one instance of the singleton class can exist.
    """
    _instances = {}
    _lock: Lock = Lock()  # Lock object to ensure thread-safe access to instances.

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

