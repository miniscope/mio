"""
CLI commands for running streamDaq
"""

import os
from collections.abc import Callable
from pathlib import Path

import click

from mio.cli.common import ConfigIDOrPath
from mio.devices.stream import StreamDevice
from mio.devices.stream.config import StreamDevConfig
from mio.models.process import FrequencyMaskingConfig
from mio.ntp import prompt_ntp_sync


@click.group()
def stream() -> None:
    """
    Command group for StreamDevice
    """
    pass


def _common_options(fn: Callable) -> Callable:
    fn = click.option(
        "-c",
        "--device_config",
        required=True,
        help=(
            "Either a config `id` or a path to device config YAML file for streamDaq. "
            "If you aren't using `id` you can ignore them."
            "(see models.stream.StreamDevConfig). If path is relative, treated as "
            "relative to the current directory, and then if no matching file is found, "
            "relative to the user `config_dir` (see `mio config --help`)."
        ),
        type=ConfigIDOrPath(),
    )(fn)
    return fn


def _capture_options(fn: Callable) -> Callable:
    fn = click.option(
        "-o",
        "--output",
        help="Output file basename for video, metadata, and binary exports",
        type=click.Path(),
    )(fn)
    fn = click.option(
        "-ok",
        "--output-kwarg",
        "okwarg",
        help="Output kwargs (passed to StreamDevice.init_video). \n"
        "passed as (potentially multiple) calls like\n\n"
        "mio stream capture -ok key1 val1 -ok key2 val2",
        multiple=True,
    )(fn)
    fn = click.option("--no-display", is_flag=True, help="Don't show video in real time")(fn)
    fn = click.option("-b", "--binary_export", is_flag=True, help="Save binary to a .bin file")(fn)
    fn = click.option(
        "-m",
        "--metadata_display",
        is_flag=True,
        help="Display metadata in real time. \n"
        "**WARNING:** This is still an **EXPERIMENTAL** feature and is **UNSTABLE**.",
    )(fn)
    fn = click.option(
        "-f",
        "--freq_mask_config",
        help="Path to, or ID of frequency masking config YAML file - "
        "applies frequency masking to the displayed video, "
        "but preserves raw video and does not modify the video output written to disk "
        "(apply postprocessing separately).",
        type=ConfigIDOrPath(),
    )(fn)
    fn = click.option(
        "--mode",
        type=click.Choice(["image", "ber"]),
        default="image",
        show_default=True,
        help="Capture mode. 'image' produces video/metadata; 'ber' runs a "
        "PRBS bit-error-rate test and produces no video/metadata output.",
    )(fn)
    return fn


@stream.command()
@_common_options
@_capture_options
def capture(
    device_config: Path,
    freq_mask_config: Path | None,
    output: Path | None,
    okwarg: dict | None,
    no_display: bool | None,
    binary_export: bool | None,
    metadata_display: bool | None,
    mode: str,
    **kwargs: dict,
) -> None:
    """
    Capture video from a StreamDevice device, optionally saving as an encoded video or as raw binary
    """

    # Rather don't like getting config here, but I want to do ntp check in the CLI so it's here.
    config = StreamDevConfig.from_any(device_config)
    if config.runtime.ntp_server is not None:
        prompt_ntp_sync(
            config.runtime.ntp_server, max_offset_seconds=config.runtime.ntp_max_offset_seconds
        )

    daq_inst = StreamDevice(device_config=device_config)
    okwargs = dict(okwarg)

    if output:
        unique_stem_path = get_unique_stempath(Path(output))
        video_output = unique_stem_path.with_suffix(".avi") if mode == "image" else None
        metadata_output = unique_stem_path.with_suffix(".csv") if mode == "image" else None
        binary_output = unique_stem_path.with_suffix(".bin") if binary_export else None
        ber_output = unique_stem_path.with_suffix(".json") if mode == "ber" else None
    else:
        video_output = None
        metadata_output = None
        binary_output = None
        ber_output = None

    if freq_mask_config:
        freq_mask_config = FrequencyMaskingConfig.from_any(freq_mask_config)
    else:
        freq_mask_config = None

    daq_inst.capture(
        source="fpga",
        video=video_output,
        video_kwargs=okwargs,
        metadata=metadata_output,
        binary=binary_output,
        show_video=not no_display and mode == "image",
        show_metadata=metadata_display and mode == "image",
        freq_mask_config=freq_mask_config,
        mode=mode,
        ber_output=ber_output,
    )


@stream.command()
@_common_options
@click.option(
    "-s",
    "--source",
    required=True,
    help="Path to RAW FPGA data to plug into okDevMock",
    type=click.Path(exists=True),
)
@click.option(
    "-p", "--profile", is_flag=True, default=False, help="Run with profiler (not implemented yet)"
)
@_capture_options
@click.pass_context
def test(ctx: click.Context, source: Path, profile: bool, **kwargs: dict) -> None:
    """
    Run StreamDevice in testing mode, using the okDevMock rather than the actual device
    """
    if profile:
        raise NotImplementedError("Profiling mode is not implemented")

    os.environ["STREAMDAQ_MOCKRUN"] = "just_placeholder"
    os.environ["PYTEST_OKDEV_DATA_FILE"] = str(source)

    ctx.forward(capture)


def get_unique_stempath(base_output: Path) -> Path:
    """
    Check the target directory if there are any files with the same basename (ignoring extensions)
    If so, append a number to the basename to make it unique.
    """
    directory = base_output.parent
    stem = base_output.stem

    # Ensure the directory exists
    directory.mkdir(parents=True, exist_ok=True)

    index = 1
    candidate_stem = stem

    def _any_stem_exists(candidate_stem_str: str) -> bool:
        # List all files and check if any have the same stem as the candidate
        return any(candidate_stem_str == p.stem for p in directory.iterdir() if p.is_file())

    # Iterate to find a unique stem
    while _any_stem_exists(candidate_stem):
        candidate_stem = f"{stem}-{index}"
        index += 1

    return directory / candidate_stem
