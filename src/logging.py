import yaml
import logging
import logging.config
import json_logging
import coloredlogs
from pathlib import Path
from typing import List
from multiprocessing import current_process

loggers = {}

class LoggerFactory:

    @classmethod
    def _get_logging_paths(cls, config: dict) -> List[Path]:
        handlers = config['handlers']
        loggin_paths = []

        for handler_name in handlers.keys():
            file_name = handlers[handler_name].get('filename')
            if file_name:
                base_path = Path(handlers[handler_name].get('filename')).parent
                loggin_paths.append(base_path)

        return list(set(loggin_paths))


    @classmethod
    def _create_logging_directories(cls, paths: List[Path]) -> None:
        for path in paths:
            if not path.exists():
                path.mkdir()


    @classmethod
    def __create_logger(cls, config_file='logger_settings.yaml', level=logging.INFO):
        json_logging.init_non_web(enable_json=True)

        config_file_path = Path.cwd().joinpath(config_file)
        if config_file_path.exists():
            with open(str(config_file_path), 'rt') as file:
                try:
                    config = yaml.safe_load(file.read())
                    cls._create_logging_directories(cls._get_logging_paths(config))
                    logging.config.dictConfig(config)
                    coloredlogs.install()
                except Exception as e:
                    print('Error in Logging Configuration. Using default configs')
                    logging.basicConfig(level=level)
                    coloredlogs.install(level=level)
        else:
            logging.basicConfig(level=level)
            coloredlogs.install(level=level)
            print('Failed to load configuration file. Using default configs')

        return logging.getLogger(__name__)

    @classmethod
    def get_logger(cls, _):
        execution_context = f'{current_process().name}'

        global loggers
        logger = loggers.get(execution_context)

        if not logger:
            logger = cls.__create_logger()
            loggers[execution_context] = logger

        return logger
