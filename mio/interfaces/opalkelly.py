"""
Interfaces for OpalKelly (model number?) FPGAs
"""

import sys

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from mio.exceptions import (
    DeviceConfigurationError,
    DeviceOpenError,
    StreamReadError,
)
from mio.logging import init_logger
from mio.vendor import opalkelly as ok


class okDev(ok.okCFrontPanel):
    """
    I/O and configuration for an (what kind of opal kelly device?)

    .. todo::

        Phil: document what this thing does, including how bitfiles work
        and how they're generated/where they're located.

    """

    def __init__(self, read_length: int, serial_id: str = ""):
        super().__init__()
        self.logger = init_logger("okDev")
        self.read_length = read_length
        ret = self.OpenBySerial("")
        if ret != self.NoError:
            raise DeviceOpenError(f"Cannot open device: {serial_id}")
        self.info = ok.okTDeviceInfo()
        ret = self.GetDeviceInfo(self.info)
        if ret == self.NoError:
            self.logger.info(f"Connected to {self.info.productName}")

    def upload_bit(self, bit_file: str) -> None:
        """
        Upload a configuration bitfile to the FPGA

        Args:
            bit_file (str): Path to the bitfile
        """

        ret = self.ConfigureFPGA(bit_file)
        if ret == self.NoError:
            self.logger.debug(f"Succesfully uploaded {bit_file}")
        else:
            raise DeviceConfigurationError(f"Configuration of {self.info.productName} failed")
        self.logger.debug(
            "FrontPanel {} supported".format("is" if self.IsFrontPanelEnabled() else "not")
        )
        ret = self.ResetFPGA()

    def read_data(
        self, length: int | None = None, addr: int = 0xA0, blockSize: int = 16
    ) -> bytearray:
        """
        Read a buffer's worth of data

        Args:
            length (int): Amount of data to read
            addr (int): FPGA address to read from
            blockSize (int): Size of individual blocks (in what unit?)

        Returns:
            :class:`bytearray`
        """
        if length is None:
            length = self.read_length
        buf = bytearray(length)
        ret = self.ReadFromBlockPipeOut(addr, data=buf, blockSize=blockSize)
        if ret < 0:
            msg = f"Read failed: {ret}"
            self.logger.error(msg)
            raise StreamReadError(msg)
        elif ret < length:
            self.logger.warning(f"Only {ret} bytes read")
        return buf

    def set_wire(self, addr: int, val: int) -> None:
        """
        .. todo::

            Phil! what does this do?

        Args:
            addr: ?
            val: ?
        """
        ret = self.SetWireInValue(addr, val)
        ret = self.UpdateWireIns()
        if ret != self.NoError:
            raise DeviceConfigurationError(f"Wire update failed: {ret}")

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> bytes:
        return self.read_data()
