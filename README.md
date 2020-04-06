My implementations of `cs231n` Spring  2019 Assignments.

Original official version of assignments without answers could be find [here](https://github.com/cs231n/cs231n.github.io/tree/master/assignments).

Most of the implementation ideas have been described in comments, and more notes will be updated one after another.

## Enviroment  Issues
- pytorch in windows

If you cannot install `pytorch` in Windows by `pip` directly, you can find how to install it in the [website of pytorch](https://pytorch.org/).

- c extension in windows

In assignment2, you need to compile some C extensions yourself before using faster_layers.It's quite troublesom for Python 3.x in Windows. It requires `VC 14.x` for compiling C extensions, so you need install Visual Studio 2017 or later, which is pretty big.

Of course, you can try to find other people's compiled version online. My compiled file for py3.6 in win64 in [here](assignment2/cs231n/im2col_cython.cp36-win_amd64.pyd).

- packages version of assignment3

The packages in `requirements.txt` of assignment3 are too old for python  3.7. Packages in`requirements.txt` of assignment2 work well for assignment3.

## Note
### matmul
The behavior depends on the arguments in the following way.

- If both arguments are 2-D they are multiplied like conventional matrices.
- If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
- If the first argument is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
- If the second argument is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.

`matmul` differs from `dot` in two important ways:

- Multiplication by scalars is not allowed, use `*` instead.

- Stacks of matrices are broadcast together as if the matrices were elements, respecting the signature `(n,k),(k,m)->(n,m)`:

```
>>> a = np.ones([9, 5, 7, 4])
>>> c = np.ones([9, 5, 4, 3])
>>> np.dot(a, c).shape
(9, 5, 7, 9, 5, 3)
>>> np.matmul(a, c).shape
(9, 5, 7, 3)
>>> # n is 7, k is 4, m is 3
```
The matmul function implements the semantics of the `@` operator introduced in Python 3.5 following PEP465. 