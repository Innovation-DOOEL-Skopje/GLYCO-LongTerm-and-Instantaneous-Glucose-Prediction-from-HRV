import sys
import importlib


#____________________________________________________________________________________________________________________________________

def reload_modules() -> None:
    """Non-functional"""

    print('not functional')

#____________________________________________________________________________________________________________________________________


def delete_local_variables() -> None:

    for name in dir():
        if not name.startswith('_'):
            del globals()[name]



