import time
import serial

with serial.Serial("/dev/ttyACM0", 115200, timeout=0.5) as s:
    s.dtr = True
    s.rts = True
    s.reset_input_buffer()
    buf = b""
    t0 = time.time()
    while time.time() - t0 < 5:
        buf += s.read(4096)
    print(len(buf), "bytes in 5 s")
    print(buf[:200].hex(" "))