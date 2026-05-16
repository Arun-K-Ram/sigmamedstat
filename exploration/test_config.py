from crip_x.utils.config import settings, ROOT_DIR
from crip_x.utils.logger import get_logger

logger = get_logger(__name__)

logger.info(f"Root dir: {ROOT_DIR}")
logger.info(f"Environment: {settings.environment}")
logger.info(f"API will run on: {settings.api_host}:{settings.api_port}")
logger.warning("This is what a warning looks like")
logger.debug("This is what a debug message looks like")