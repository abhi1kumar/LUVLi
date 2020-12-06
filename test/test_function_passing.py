
import argparse

def a(i):
    return i + 0
def b(i):
    return i + 1
def c(i):
    return i + 2

# a dictionary mapping strings of function names to function objects:
funcs = {'0': a, '1': b, '2': c}

parser = argparse.ArgumentParser()
# Add the -f/--func argument: valid choices are function _names_
parser.add_argument('-f', '--func', default='1',
                     help="""Choose one of the specified function to be run.""")

args = parser.parse_args()

# Resolve the chosen function object using its name:    
chosen_func = funcs[args.func]

print(chosen_func(10))
