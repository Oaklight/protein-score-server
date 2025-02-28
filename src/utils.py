from typing import Any, Optional
import logging
import logging.config

logger = logging.getLogger(__name__)

# ANSI escape codes for colors
COLOR_MAPPING = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_black": "\033[90m",
    "bright_red": "\033[91m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_blue": "\033[94m",
    "bright_magenta": "\033[95m",
    "bright_cyan": "\033[96m",
    "bright_white": "\033[97m",
}

# ANSI escape code for bold text
BOLD = "\033[1m"
# ANSI escape code to reset all styles
RESET = "\033[0m"


def colorstring(
    message: Any,  # Accept any type of input
    color: Optional[str] = "green",
    bold: bool = False,
) -> str:
    """
    Returns a colored string using ANSI escape codes.

    :param message: The message to be colored. Can be of any type (e.g., str, int, float, bool).
    :param color: The color to apply. Supported colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white',
                 and their bright variants: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow', 'bright_blue',
                 'bright_magenta', 'bright_cyan', 'bright_white'.
    :param bold: If True, applies bold styling to the text.
    :return: A string with the specified color and styling.
    """
    # Convert the message to a string
    message_str = str(message)

    # Get the ANSI color code
    color_code = COLOR_MAPPING.get(color.lower(), COLOR_MAPPING["white"])

    # Apply color and bold styling
    styled_message = f"{color_code}{message_str}{RESET}"
    if bold:
        styled_message = f"{BOLD}{styled_message}"

    return styled_message


def cprint(message: Any, color: Optional[str] = "green", **kwargs) -> None:
    """
    Prints a colored string using ANSI escape codes.

    :param message: The message to be colored. Can be of any type (e.g., str, int, float, bool).
    :param color: The color to apply. Supported colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'.
    :param kwargs: Additional keyword arguments to pass to the `print` function (e.g., `end`, `sep`, `file`, `flush`).
    """
    print(colorstring(message, color), **kwargs)
