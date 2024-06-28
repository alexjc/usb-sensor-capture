#
# Copyright (c) 2024, Alex J. Champandard.
#

import math
import time
import queue
import logging
import threading

import numpy as np
import usb.core as usb


log = logging.getLogger('usb')


CDC_CMDS = {
    "SEND_ENCAPSULATED_COMMAND": 0x00,
    "GET_ENCAPSULATED_RESPONSE": 0x01,
    "SET_COMM_FEATURE": 0x02,
    "GET_COMM_FEATURE": 0x03,
    "CLEAR_COMM_FEATURE": 0x04,
    "SET_LINE_CODING": 0x20,
    "GET_LINE_CODING": 0x21,
    "SET_CONTROL_LINE_STATE": 0x22,
    "SEND_BREAK": 0x23,
}


class ComPort(object):
    def __init__(self, usb_device, start=True, frequency=540):
        self.device = usb_device
        self._isFTDI = False
        self._rxqueue = queue.Queue(maxsize=3)
        self._rxthread = None
        self._rxactive = False
        self.baudrate = 9600
        self.parity = 0
        self.stopbits = 1
        self.databits = 8

        self.time_period = 1.0 / frequency
        if not isinstance(start, bool):
            self.time_start = start
        else:
            self.time_start = time.time()

        cfg = usb_device.get_active_configuration()

        data_itfs = list(
            usb.util.find_descriptor(
                cfg, find_all=True, custom_match=lambda e: (e.bInterfaceClass == 0xA)
            )
        )
        if not data_itfs:
            print("Unable to connect.  No data interfaces on device")
            exit()

        data_itf = data_itfs[0]
        cmd_itfs = list(
            usb.util.find_descriptor(
                cfg, find_all=True, custom_match=lambda e: (e.bInterfaceClass == 0x2)
            )
        )
        itf_num = cmd_itfs[0].bInterfaceNumber
        assert len(cmd_itfs) == len(data_itfs), "COM port data / command interface mismatch"

        # ports = len(data_itfs)
        self._ep_in = usb.util.find_descriptor(
            data_itf, custom_match=lambda e: (e.bEndpointAddress & 0x80)
        )
        assert self._ep_in.wMaxPacketSize == 64
        self._ep_out = usb.util.find_descriptor(
            data_itf, custom_match=lambda e: not (e.bEndpointAddress & 0x80)
        )

        self._startRx()

    def _startRx(self):
        if self._rxthread is not None and (self._rxactive or self._rxthread.isAlive()):
            return
        self._rxactive = True
        self._rxthread = threading.Thread(target=self._read_sample)
        self._rxthread.daemon = True
        self._rxthread.start()

    def _endRx(self):
        self._rxactive = False

    def _read_sample(self):
        current = time.time()
        sync = math.fmod(current - self.time_start, self.time_period)
        while sync < 0.0: sync += self.time_period
        if sync > 0.0: time.sleep(sync)

        while True:
            try:
                while self._rxqueue.full():
                    time.sleep(0.0)

                rv = self._ep_in.read(self._ep_in.wMaxPacketSize, timeout=1000)
                assert len(rv) == 64
                self._rxqueue.put(rv)
            except usb.USBError as e:
                log.warning(f"USB Error on _read {e}, packets: {self._ep_in.wMaxPacketSize}")
                break

            current += self.time_period
            remaining = max(current - time.time(), 0.0)
            if remaining > 0.0:
                time.sleep(remaining)

    def _read_blocking(self):
        """ check ep for data, add it to queue and sleep for interval """
        while self._rxactive:
            try:
                while not self._rxqueue.empty():
                    time.sleep(0.0)

                rv = self._ep_in.read(self._ep_in.wMaxPacketSize, timeout=1000)
                assert len(rv) == 64
                self._rxqueue.put(rv)
            except usb.USBError as e:
                log.warning(
                    f"USB Error on _read {e}, packets: {self._ep_in.wMaxPacketSize}"
                )
                break

    def _getRxLen(self):
        return self._rxqueue.qsize()

    rxlen = property(fget=_getRxLen)

    def readBytes(self, size=None):
        while not self._rxqueue.empty():
            rx = self._rxqueue.get()
            if size is not None:
                assert len(rx) == size
            return rx

    def readText(self):
        return "".join(chr(c) for c in self.readBytes())

    def write(self, data):
        try:
            ret = self._ep_out.write(data)
        except usb.USBError as e:
            log.warning("USB Error on write {}".format(e))
            return

        if len(data) != ret:
            log.error("Bytes written mismatch {0} vs {1}".format(len(data), ret))
        else:
            log.debug("{} bytes written to ep".format(ret))

    def setControlLineState(self, rts=None, dtr=None):
        ctrlstate = (2 if rts else 0) + (1 if dtr else 0)
        if self._isFTDI:
            ctrlstate += (1 << 8) if dtr is not None else 0
            ctrlstate += (2 << 8) if rts is not None else 0

        # 0:OUT, 1:IN
        txdir = 0
        # 0:std, 1:class, 2:vendor
        req_type = 2 if self._isFTDI else 1
        # 0:device, 1:interface, 2:endpoint, 3:other
        recipient = 0 if self._isFTDI else 1
        req_type = (txdir << 7) + (req_type << 5) + recipient

        wlen = self.device.ctrl_transfer(
            bmRequestType=req_type,
            bRequest=1 if self._isFTDI else CDC_CMDS["SET_CONTROL_LINE_STATE"],
            wValue=ctrlstate,
            wIndex=1 if self._isFTDI else 0,
            data_or_wLength=0,
        )

    def setLineCoding(self, baudrate=None, parity=None, databits=None, stopbits=None):
        sbits = {1: 0, 1.5: 1, 2: 2}
        dbits = {5, 6, 7, 8, 16}
        pmodes = {0, 1, 2, 3, 4}
        brates = {
            300,
            600,
            1200,
            2400,
            4800,
            9600,
            14400,
            19200,
            28800,
            38400,
            57600,
            115200,
            230400,
        }

        if stopbits is not None:
            if stopbits not in sbits.keys():
                valid = ", ".join(str(k) for k in sorted(sbits.keys()))
                raise ValueError("Valid stopbits are " + valid)
            self.stopbits = stopbits

        if databits is not None:
            if databits not in dbits:
                valid = ", ".join(str(d) for d in sorted(dbits))
                raise ValueError("Valid databits are " + valid)
            self.databits = databits

        if parity is not None:
            if parity not in pmodes:
                valid = ", ".join(str(pm) for pm in sorted(pmodes))
                raise ValueError("Valid parity modes are " + valid)
            self.parity = parity

        if baudrate is not None:
            if baudrate not in brates:
                brs = sorted(brates)
                dif = [abs(br - baudrate) for br in brs]
                best = brs[dif.index(min(dif))]
                raise ValueError("Invalid baudrates, nearest valid is {}".format(best))
            self.baudrate = baudrate

        if self._isFTDI:
            self._setBaudFTDI(self.baudrate)
            self._setLineCodeFTDI(
                bits=self.databits,
                stopbits=sbits[self.stopbits],
                parity=self.parity,
                breaktype=0,
            )
        else:
            linecode = [
                self.baudrate & 0xFF,
                (self.baudrate >> 8) & 0xFF,
                (self.baudrate >> 16) & 0xFF,
                (self.baudrate >> 24) & 0xFF,
                sbits[self.stopbits],
                self.parity,
                self.databits,
            ]

            txdir = 0  # 0:OUT, 1:IN
            req_type = 1  # 0:std, 1:class, 2:vendor
            recipient = 1  # 0:device, 1:interface, 2:endpoint, 3:other
            req_type = (txdir << 7) + (req_type << 5) + recipient

            wlen = self.device.ctrl_transfer(
                req_type, CDC_CMDS["SET_LINE_CODING"], data_or_wLength=linecode
            )

    def getLineCoding(self):
        if self._isFTDI:
            log.warning("FTDI does not support reading baud parameters")
        txdir = 1  # 0:OUT, 1:IN
        req_type = 1  # 0:std, 1:class, 2:vendor
        recipient = 1  # 0:device, 1:interface, 2:endpoint, 3:other
        req_type = (txdir << 7) + (req_type << 5) + recipient

        buf = self.device.ctrl_transfer(
            bmRequestType=req_type,
            bRequest=CDC_CMDS["GET_LINE_CODING"],
            wValue=0,
            wIndex=0,
            data_or_wLength=255,
        )
        self.baudrate = buf[0] + (buf[1] << 8) + (buf[2] << 16) + (buf[3] << 24)
        self.stopbits = 1 + (buf[4] / 2.0)
        self.parity = buf[5]
        self.databits = buf[6]
        # print("LINE CODING:")
        # print("  {0} baud, parity mode {1}".format(self.baudrate, self.parity))
        # print("  {0} data bits, {1} stop bits".format(self.databits, self.stopbits))
        assert self.parity == 2  # Raw Data (ADC Samples)

    def disconnect(self):
        self._endRx()
        while self._rxthread is not None and self._rxthread.isAlive():
            pass
        usb.util.dispose_resources(self.device)
        if self._rxthread is None:
            log.debug("Rx thread never existed")
        else:
            log.debug(
                "Rx thread is {}".format(
                    "alive" if self._rxthread.isAlive() else "dead"
                )
            )
        attempt = 1
        while attempt < 10:
            try:
                self.device.attach_kernel_driver(0)
                log.debug("Attach kernal driver on attempt {0}".format(attempt))
                break
            except usb.USBError:
                attempt += 1
                time.sleep(0.1)
        if attempt == 10:
            log.error("Could not attach kernal driver")


def selectDevice():
    devices = [d for d in usb.find(find_all=True) if d.bDeviceClass in {0, 2, 0xFF}]
    return devices


class GatheringThread(threading.Thread):
    def __init__(self, samples=64, frequency=540.0, duration=150.0):
        super(GatheringThread, self).__init__(target=self.main)

        self.samples = samples
        self.frequency = frequency
        self.frame = 0
        self._active = True

        self.current_targets = None

        ports = selectDevice()
        start = time.time()
        delay = 1.0 / (len(ports) * frequency)

        self.ports = []
        for i, d in enumerate(ports):
            p = ComPort(d, start=start + i * delay, frequency=frequency)
            p.setControlLineState(rts=True, dtr=True)
            p.setLineCoding(baudrate=230400, stopbits=1, parity=2, databits=0x08)
            p.getLineCoding()
            self.ports.append(p)
        assert len(self.ports) > 0, "No USB devices found."

        total = int(frequency * duration)
        self.data_inputs = np.zeros((total, len(self.ports), 16, 4), dtype=np.uint8)
        self.data_targets = np.zeros((total, 4), dtype=np.uint8)

    def main(self):
        samples = np.zeros((len(self.ports), self.samples), dtype=np.uint8)
        start = time.time()
        current = start

        print('Capture thread starting...')

        while self._active is True:
            # Sample data from sensors to match target.
            for i, p in enumerate(self.ports):
                while p.rxlen == 0:
                    time.sleep(0.0)

                data = p.readBytes(size=self.samples)
                assert len(data) == self.samples
                samples[i] = data

            current += 1.0 / self.frequency
            remaining = max(current - time.time(), 0.0)
            if remaining > 0.0:
                time.sleep(remaining)

            if self.frame % 2_700 == 0:
                print(self.frame / (time.time() - start), 'Hz')

            # Store the sampled sensor data.
            self.data_inputs[self.frame] = samples.reshape(self.data_inputs.shape[1:])
            self.data_targets[self.frame] = self.current_targets
            self.frame += 1

            if self.frame >= self.data_inputs.shape[0]:
                print('Capture duration elapsed!')
                break

        timestamp = int(start)
        if not self._active:
            print('Capture thread terminated!', int(timestamp))
        self._active = False

        # print(' - targets min/max', self.data_targets.min(axis=0), self.data_targets.max(axis=0))
        # print(' - data    min/max', self.data_inputs.min(axis=(0,1)), self.data_inputs.max(axis=(0,1)), sep='\n')

        with open('data/capture-%i.npy' % timestamp, 'wb') as f:
            np.save(f, self.data_inputs)
            np.save(f, self.data_targets)


    def stop(self):
        self._active = False
