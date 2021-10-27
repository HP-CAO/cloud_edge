import concurrent.futures

with concurrent.futures.ProcessPoolExecutor() as executor:
    pass

with concurrent.futures.ThreadPoolExecutor() as executor_1:
    pass

from multiprocessing import shared_memory

