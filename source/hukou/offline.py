"""No Python network transport is permitted inside the application."""
import socket

def install_guard() -> None:
    def denied(*args,**kwargs):
        raise OSError('此程序仅允许本地离线处理，不支持网络连接')
    socket.socket.connect=denied
    socket.socket.connect_ex=denied
    socket.create_connection=denied
