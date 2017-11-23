import logging
import sys

from allennlp.commands import main

from dataset_readers import *  # noqa: F401,F403
from models import *  # noqa: F401,F403

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

if __name__ == '__main__':
    main(prog=sys.argv[0])
