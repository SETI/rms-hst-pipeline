import logging


def init_logging() -> None:
    """
    A central place for configuring logging.
    """
    logging.basicConfig(
        format="[%(levelname)s %(asctime)s %(name)r]: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.DEBUG,
    )
