"""
Bit operations for parsing header and payload information. Currently for use in streamDaq module.
"""


import numpy as np


class BufferFormatter:
    """
    Class for bit/byte operations and converting the input buffer into header and data numpy arrays.
    """

    @staticmethod
    def _reverse_bits_in_array(arr: np.ndarray) -> np.ndarray:
        """
        Reverse the bit order in each 32-bit element of a numpy array.
        Check out the input/output in binary format to see the bit orders are reversed.

        Parameters
        ----------
        arr : np.ndarray
            The input array.

        Returns
        -------
        np.ndarray
            The array with bits in each element reversed.

        Examples
        --------
        >>> _reverse_bits_in_array(np.array([0b11000011101010000100000000000000], dtype=np.uint32))
        array([136643], dtype=uint32) #0b00000000000000100001010111000011 in binary format
        >>> _reverse_bits_in_array(np.array([0b00000000000001111000000000000111], dtype=np.uint32))
        array([3758219264], dtype=uint32) #0b11100000000000011110000000000000 in binary format
        """
        arr = np.unpackbits(arr.view(np.uint8)).reshape(-1, 32)[:, ::-1]
        return np.packbits(arr).view(np.uint32)

    @staticmethod
    def _reverse_byte_order_in_array(arr: np.ndarray) -> np.ndarray:
        """
        Reverse the byte (8-bit) order of each 32-bit element in a numpy array.
        Check out the input/output in hexadecimal format.
        The order of each byte (2-digits in hex) in each 32-bit element is reversed.

        Parameters
        ----------
        arr : np.ndarray
            The input array.

        Returns
        -------
        np.ndarray
            The array with byte order in each element reversed.

        Examples
        --------
        >>> out = _reverse_byte_order_in_array(np.array([0x12345678, 0x11002200], dtype=np.uint32))
        >>> np.vectorize(lambda x: f"0x{x:08X}")(out)
        array(['0x78563412', '0x00220011'], dtype='<U10')
        """
        return arr.byteswap()

    @classmethod
    def bytebuffer_to_ndarrays(
        cls,
        buffer: bytes,
        header_length_words: int,
        preamble_length_words: int,
        reverse_header_bits: bool,
        reverse_header_bytes: bool,
        reverse_payload_bits: bool,
        reverse_payload_bytes: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Format the buffer and return the header (uint32) and payload (uint8) arrays.
        The bit/byte order can be optionally reversed.

        The buffer is a concat of a uint32 header array and a uint8 payload array in bytes format.
        The inconsistency in element size arises from how headers are formatted in the MCU.
        Each sensor data is uint8, while metadata (header) uses uint32 because it needs larger size.

        Bit/byte reversing is required because we handle all of this as a uint32 array in the MCU.
        This adds complexity to doing the bit/byte ordering on the MCU so we convert it here.

        Parameters
        ----------
        buffer : bytes
            The binary buffer to format.
        header_length_words : int
            Length of the header in words.
        preamble_length_words : int
            Length of the preamble in words.
        reverse_header_bits : bool
            Reverse bits in the header if True. Default is True.
        reverse_header_bytes : bool
            Reverse byte order in the header if True. Default is True.
        reverse_payload_bits : bool
            Reverse bits in the body if True. Default is True.
        reverse_payload_bytes : bool
            Reverse byte order in the body if True. Default is True.

        Returns
        -------
        tuple
            A tuple containing the processed header as a numpy array of uint32
            and the processed payload as a numpy array of uint8.

        Examples
        --------
        Inputs like::

            buf=b'\\x12\\x34\\x56\\x78\\x11\\x11\\x11\\x11\\x00\\x11\\x22\\x33\\x44\\x55\\x66\\x77',
            header_length_words=1,
            preamble_length_words=1,
            reverse_header_bits=False,
            reverse_header_bytes=False,
            reverse_payload_bits=False,
            reverse_payload_bytes=False,

        Will return the following header array (uint32) and payload array (uint8)::

            np.array([0x11111111], dtype=np.uint32),
            np.array([0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77], dtype=np.uint8)

        Note that this truncates the preamble from the header.
        The first word :code:`0x12345678` is truncated because :code:`preamble_length_words=1`.
        """

        # Padding in case the buffer is not a multiple of 4 bytes (32 bits)
        padding_length = (4 - (len(buffer) % 4)) % 4
        padded_buffer = buffer + b"\x00" * padding_length

        # Convert the padded buffer to a uint32 numpy array
        data = np.frombuffer(padded_buffer, dtype=np.uint32)

        # Process header
        header = data[preamble_length_words:header_length_words]
        if reverse_header_bits:
            header = cls._reverse_bits_in_array(header)
        if reverse_header_bytes:
            header = cls._reverse_byte_order_in_array(header)

        # Process body
        payload_data = data[header_length_words:]
        if reverse_payload_bits:
            payload_data = cls._reverse_bits_in_array(payload_data)
        if reverse_payload_bytes:
            payload_data = cls._reverse_byte_order_in_array(payload_data)

        # Convert processed body buffer to uint8 numpy array
        payload_uint8 = payload_data.view(np.uint8)
        return header, payload_uint8
