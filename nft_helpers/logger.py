# Create logger object
import logging


def create_logger(name, level='DEBUG', mode='w', filename='log.log'):
    """Get a default logger using a provided name"""
    logger = logging.getLogger(name)
    handler = logging.FileHandler(filename, mode=mode)
    formatter = logging.Formatter('%(asctime)s - %(levelname) s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    
    return logger