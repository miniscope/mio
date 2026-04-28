import numpy as np
import pytest

from mio.bit_operation import BufferFormatter


@pytest.mark.parametrize(
    "test_input,header_length_words,preamble_length_words,reverse_header_bits,reverse_header_bytes,reverse_payload_bits,reverse_payload_bytes,expected_header,expected_payload",
    [
        (
            b"\x12\x34\x56\x78\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb",
            1,
            0,
            False,
            False,
            False,
            False,
            np.array([0x78563412], dtype=np.uint32),
            np.array(
                [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB],
                dtype=np.uint8,
            ),
        ),
        (
            b"\x12\x34\x56\x78\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb",
            1,
            0,
            True,
            True,
            True,
            True,
            np.array([0x1E6A2C48], dtype=np.uint32),
            np.array(
                [0x00, 0x88, 0x44, 0xCC, 0x22, 0xAA, 0x66, 0xEE, 0x11, 0x99, 0x55, 0xDD],
                dtype=np.uint8,
            ),
        ),
    ],
)
def test_bytebuffer_to_ndarrays(
    test_input,
    header_length_words,
    preamble_length_words,
    reverse_header_bits,
    reverse_header_bytes,
    reverse_payload_bits,
    reverse_payload_bytes,
    expected_header,
    expected_payload,
):
    header, payload = BufferFormatter.bytebuffer_to_ndarrays(
        test_input,
        header_length_words,
        preamble_length_words,
        reverse_header_bits,
        reverse_header_bytes,
        reverse_payload_bits,
        reverse_payload_bytes,
    )
    """
    Test for ensuring that the conversion and optional bit/byte reversals are performed correctly.
    The buffer is a concat of a 32-bit array for metadata and an 8-bit array for payload.
    The input/output in the reverseed bit/bytes is non-intuitive so don't think what it means.
    """
    np.testing.assert_array_equal(header, expected_header)
    np.testing.assert_array_equal(payload, expected_payload)


@pytest.mark.parametrize(
    "input_array,expected_output",
    [
        (
            np.array([0b11000011101010000100000000000000], dtype=np.uint32),
            np.array([0b00000000000000100001010111000011], dtype=np.uint32),
        ),
        (
            np.array([0b00000000000001111000000000000111], dtype=np.uint32),
            np.array([0b11100000000000011110000000000000], dtype=np.uint32),
        ),
        (
            np.array([0b10101010101010101010101010101010], dtype=np.uint32),
            np.array([0b01010101010101010101010101010101], dtype=np.uint32),
        ),
    ],
)
def test_reverse_bits_in_array(input_array, expected_output):
    """
    Test for flipping bit order in a 32-bit word.
    """
    result = BufferFormatter._reverse_bits_in_array(input_array)
    np.testing.assert_array_equal(result, expected_output)


@pytest.mark.parametrize(
    "input_array,expected_output",
    [
        (np.array([0x12345678], dtype=np.uint32), np.array([0x78563412], dtype=np.uint32)),
        (np.array([0xABCD4574], dtype=np.uint32), np.array([0x7445CDAB], dtype=np.uint32)),
        (np.array([0x11002200], dtype=np.uint32), np.array([0x00220011], dtype=np.uint32)),
    ],
)
def test_reverse_byte_order_in_array(input_array, expected_output):
    """
    Test for flipping byte order (8-bit chunks) in a 32-bit word.
    """
    result = BufferFormatter._reverse_byte_order_in_array(input_array)
    np.testing.assert_array_equal(result, expected_output)
