"""Bitwise error rate calculation"""

import logging
import multiprocessing

import numpy as np

from mio.devices.stream import StreamBufferHeader, StreamDevConfig
from mio.devices.stream.nodes import exact_iter


def prbs15_ber(
    serial_buffer_queue: multiprocessing.Queue,
    config: StreamDevConfig,
    logger: logging.Logger,
    n_buffers: int = 100,
    header_cls: type[StreamBufferHeader] = StreamBufferHeader,
) -> dict[str, float | int]:
    """
    Measure bit-error-rate (BER) and packet-error-rate (PER) on the communication link
    using PRBS-15 (pseudo-random binary sequence; standard pattern for link tests).

    Unlike typical continuous-stream BER tests, this preserves the buffer framing of
    normal image capture and substitutes PRBS-15 for the pixel payload, so the
    same data path that delivers images is what's being measured.

    Each buffer's payload is seeded with the device's buffer_count (mod 2^15,
    zero remapped to 1). The host regenerates the matching sequence and XORs
    it against the first ``pixel_count`` bytes of payload; trailing bytes
    (dummies, merged-buffer tails) are excluded so they don't inflate the count.
    Errors and bits accumulate across up to ``n_buffers`` buffers.

    PER is computed from the buffer_count range observed: any gap between the first
    and last buffer_count (beyond what was received) is treated as a dropped packet,
    and any received buffer with non-zero errors counts as an errored packet.

    KeyboardInterrupt during the run returns the partial result with ``aborted``
    set to ``"interrupted"``.

    Returns
    -------
    dict
        Run summary: ``buffers`` received, ``bits`` and ``errors`` compared,
        cumulative ``ber``, ``per`` with packet counts, ``buffer_count`` range,
        per-window snapshots, and ``aborted`` reason if the run did not complete.
    """

    def prbs15_bytes(n: int, seed: int) -> bytes:
        # PRBS-15: x^15 + x^14 + 1, MSB-first
        out = bytearray(n)
        for i in range(n):
            b = 0
            for _ in range(8):
                newbit = ((seed >> 14) ^ (seed >> 13)) & 1
                seed = ((seed << 1) | newbit) & 0x7FFF
                b = (b << 1) | (seed & 1)
            out[i] = b
        return bytes(out)

    total_bits = 0
    total_errors = 0
    total_errored_buffers = 0
    window_bits = 0
    window_errors = 0
    got = 0
    log_every = 100
    windows: list[dict[str, float | int]] = []
    first_buffer_count: int | None = None
    last_buffer_count: int | None = None
    window_first_buffer_count: int | None = None
    aborted: str | None = None

    logger.info(f"BER capture starting, target={n_buffers} buffers")

    try:
        for buf in exact_iter(serial_buffer_queue.get, None):
            header_data, payload_u8 = header_cls.from_buffer(buf, config)
            if payload_u8.size == 0:
                continue

            # Trim to pixel_count; bytes beyond are dummies or random artifacts, not PRBS.
            n_prbs = header_data.pixel_count
            if n_prbs <= 0 or n_prbs > payload_u8.size:
                continue
            payload_u8 = payload_u8[:n_prbs]

            # buffer_count seeds the PRBS.
            buffer_count = header_data.buffer_count
            seed = (buffer_count & 0x7FFF) or 1
            exp = np.frombuffer(prbs15_bytes(payload_u8.size, seed), dtype=np.uint8)

            diff = np.bitwise_xor(payload_u8, exp)
            errors = int(np.unpackbits(diff).sum())
            bits = payload_u8.size * 8

            total_errors += errors
            total_bits += bits
            if errors > 0:
                total_errored_buffers += 1
            window_errors += errors
            window_bits += bits
            got += 1
            if first_buffer_count is None:
                first_buffer_count = buffer_count
            if window_first_buffer_count is None:
                window_first_buffer_count = buffer_count
            last_buffer_count = buffer_count

            if got % log_every == 0:
                running = total_errors / total_bits if total_bits else float("nan")
                window_ber = window_errors / window_bits if window_bits else float("nan")
                logger.info(
                    f"BER progress: {got}/{n_buffers} buffers, "
                    f"bits={total_bits} errors={total_errors} "
                    f"cumulative_ber={running:.6g} window_ber={window_ber:.6g}"
                )
                windows.append(
                    {
                        "buffer_index_end": got,
                        "buffer_count_start": window_first_buffer_count,
                        "buffer_count_end": buffer_count,
                        "bits": window_bits,
                        "errors": window_errors,
                        "window_ber": window_ber,
                        "cumulative_ber": running,
                    }
                )
                window_errors = 0
                window_bits = 0
                window_first_buffer_count = None

            if got >= n_buffers:
                break
    except KeyboardInterrupt:
        logger.warning(f"BER capture interrupted by user (got {got}/{n_buffers} buffers)")
        aborted = "interrupted"

    ber = (total_errors / total_bits) if total_bits else float("nan")
    if first_buffer_count is not None and last_buffer_count is not None and got > 0:
        expected_buffers = last_buffer_count - first_buffer_count + 1
        dropped_buffers = max(expected_buffers - got, 0)
        per = (
            (dropped_buffers + total_errored_buffers) / expected_buffers
            if expected_buffers
            else float("nan")
        )
    else:
        expected_buffers = 0
        dropped_buffers = 0
        per = float("nan")

    return {
        "buffers": got,
        "bits": total_bits,
        "errors": total_errors,
        "errored_buffers": total_errored_buffers,
        "ber": ber,
        "buffer_count_start": first_buffer_count,
        "buffer_count_end": last_buffer_count,
        "expected_buffers": expected_buffers,
        "dropped_buffers": dropped_buffers,
        "per": per,
        "aborted": aborted,
        "windows": windows,
    }
