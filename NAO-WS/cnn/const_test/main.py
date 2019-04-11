from const_test.const import Const
from const_test.print import print_a

if __name__ == '__main__':
    print_a()
    Const.a = 3
    print_a()
    Const.set_a(4)
    print_a()
