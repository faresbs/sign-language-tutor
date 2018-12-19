# standard libraries
import json

class Arithmetic(object):
    """
    Used for Addition, Subtraction, Multiplication and Division Questions
    """
    def __init__(self, min_num=0, max_num=10):
        """
        :param min_num: Minimum number to start with.
        :param max_num: Maximum number to start with.
        """
        self.max_num = max_num
        self.min_num = min_num
        self.operation = None



json_settings = json.dumps([
    {
        "type": "numeric",
        "title": "Lower Limit",
        "desc": "Lowest number to be used when asking questions",
        "section": "General",
        "key": "lower_num"
    },
    {
        "type": "numeric",
        "title": "Upper Limit",
        "desc": "Highest number to be used when asking questions",
        "section": "General",
        "key": "upper_num"
    }
])