"""Generic script equivalence for comparing OCR candidates, never filling text."""
import ctypes
import os

def comparison_key(value: str) -> str:
    """Windows' language table groups traditional/simplified observed candidates."""
    if os.name!='nt':return value
    fn=ctypes.windll.kernel32.LCMapStringEx
    fn.argtypes=[ctypes.c_wchar_p,ctypes.c_uint,ctypes.c_wchar_p,ctypes.c_int,ctypes.c_wchar_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_longlong]
    fn.restype=ctypes.c_int
    target=ctypes.create_unicode_buffer(max(1,len(value)*2+1))
    count=fn('zh-CN',0x02000000,value,len(value),target,len(target),None,None,0)
    return target.value if count else value
