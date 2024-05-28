from mpi4py import MPI
import logging


class Logger():
    def __init__(self):
        comm = MPI.COMM_WORLD
        my_rank = comm.Get_rank()
        rank_count = comm.Get_size()
        logger = logging.getLogger(str(my_rank))
        logger.level = logging.INFO
        is_mpi = rank_count > 1
        if is_mpi > 0:
            logging.basicConfig(format='%(name)s: %(message)s ')
        else:
            logging.basicConfig(format='%(message)s')

        self.logger = logger
        self.is_mpi = is_mpi

    def info(self, message):
        if self.logger.level <= logging.INFO:
            if self.is_mpi:
                self.logger.info(message)
            else:
                print(message)

    def warning(self, message):
        if self.logger.level <= logging.WARNING:
            if self.is_mpi:
                self.logger.warning(message)
            else:
                print(message)

    def error(self, message):
        if self.logger.level <= logging.ERROR:
            if self.is_mpi:
                self.logger.error(message)
            else:
                print(message)


instance = Logger()
