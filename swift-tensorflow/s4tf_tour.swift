
print("Hello, Swift!")


import TensorFlow
// set PYTHON_LIBRARY -- e.g export PYTHON_LIBRARY="~/anaconda3/lib/libpython3.7m.so"
import Python

print(Python.version)

let np = Python.import("numpy")
print(np)