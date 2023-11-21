import os
from pyment import PyComment


'''
Normalize parameters value according to defined space:
    - 'quniform': round param value with defined step
    - 'constant': replace parameter's value with defined constant

:param params: Parameters
:type params: dict
:param space: Boundaries
:type space: dict
:returns: Normalized parameters
:rtype: dict
'''
filename = 'test.py'
help(PyComment)
c = PyComment(filename)
c.proceed()
c.write_patch_file()
for s in c.get_output_docs():
    print(s)