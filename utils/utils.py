import os
from dotenv import load_dotenv


def initialize_env_variables(project_root=None):
    """
    Initialize environment variables from a .env file.

    Args:
        project_root (str, optional): The root directory of the
        project. Default is None.
    """
    # Construct the full path to the .env file
    if not project_root:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

    dotenv_path = os.path.join(project_root, ".env")

    print("Initialized environment variables listed in:", dotenv_path)
    # Load the .env file
    load_dotenv(dotenv_path)