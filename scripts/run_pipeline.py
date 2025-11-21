from scripts.fetch_statcan import run as fetch_run
from scripts.transform import run as transform_run

if __name__ == "__main__":
    fetch_run()
    transform_run()
