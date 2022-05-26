import logging
from hello.src import DataClass

data = DataClass.Parameters()

print("test " + data.test_file)

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    filename = "logs.log",
    level = logging.INFO
)
global logger

logger = logging.getLogger()

logger.info("DASDAS")


print("DONE")