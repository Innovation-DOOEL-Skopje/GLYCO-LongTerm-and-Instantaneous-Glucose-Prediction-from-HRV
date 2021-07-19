import re
from typing import List

# class MyList():
#
#     my_list: List
#
#     def __init__(self):



def sortl_list_of_strings_alphanumerically(list_of_strings: List[str]) -> list:

    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split(r'([-+]?[0-9]*\.?[0-9]*)', key)]
    list_of_strings.sort(key=alphanum, reverse = True)
    list_of_strings.reverse()

    return list_of_strings



































