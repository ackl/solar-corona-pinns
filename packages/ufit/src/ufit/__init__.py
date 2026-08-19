"""UFiT Fortran field-line tracer, compiled from the pinned upstream checkout.

The wheel produced by this package contains the CFIT-patched Python wrapper
and the gfortran-built ``UFiT_Python_Callable.so`` shared library. Loading this
package is side-effect free; the tracer wrapper is imported separately by the
Grad--Rubin adapter.
"""

__version__ = "0.1.0"
