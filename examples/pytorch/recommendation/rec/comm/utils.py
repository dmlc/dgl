import io
import socket
import errno
import time

def recvall(s, n, return_fd):
    bio = io.BytesIO()
    while bio.tell() < n:
        try:
            packet = s.recv(n - bio.tell())
        except socket.error as e:
            err = e.args[0]
            if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
                time.sleep(0.05)
                continue
            else:
                raise e
        if not packet:
            raise IOError('Expected %d bytes, got %d' % (n, bio.tell()))
        bio.write(packet)
    if not return_fd:
        buf = bio.getvalue()
        bio.close()
        return buf
    else:
        bio.seek(0)
        return bio

def sendall(s, buf):
    bytes_sent = 0
    n = len(buf)
    while bytes_sent < n:
        try:
            sent = s.send(buf[bytes_sent:])
            bytes_sent += sent
        except socket.error as e:
            err = e.args[0]
            if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
                continue
            else:
                raise e
