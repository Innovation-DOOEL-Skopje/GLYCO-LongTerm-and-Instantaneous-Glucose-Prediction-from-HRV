
def print_debug(var, var_string: str = 'some var'):

    print('______________________________________')
    print(var)
    print(type(var))
    print('______________________________________')


def line(num: int = 1, half: bool = False) -> None:

    for i in range(num):
        if half:
            print('_________________________')
        else:
            print('_____________________________________________________')