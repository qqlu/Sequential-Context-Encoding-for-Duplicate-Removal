import logging
import sys
def solver_log(log_info_path):
	logger = logging.getLogger("solver")

	formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

	file_handler = logging.FileHandler(log_info_path, mode='w')
	file_handler.setFormatter(formatter)
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.formatter = formatter

	logger.addHandler(file_handler)
	logger.addHandler(console_handler)
	logger.setLevel(logging.INFO)
	return logger