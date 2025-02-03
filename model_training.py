import os

import turboml as tb
from loguru import logger

logger.info('Connecting to TurboML..')
tb.init(
  backend_url=os.environ['TURBOML_BACKEND_URL'],
  api_key=os.environ['TURBOML_API_KEY']
)
logger.info('Successfully connected to TurboML!')