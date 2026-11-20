
from src.exception import (
    DatasetNotFoundException,
    handle_exception
)

try:
    raise FileNotFoundError("Training folder missing")

except FileNotFoundError as error:
    handle_exception(
        error,
        "Unable to locate the MRI dataset"
    )
