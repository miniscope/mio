# Changelog

## 0.9

### 0.9.0 - 2026-01-27 - Batch device update, NTP sync, driver import fix

#### New features
- [`#140`](https://github.com/miniscope/mio/pull/140) - Batch upload device parameters via CLI (`mio update -b`)
- [`#142`](https://github.com/miniscope/mio/pull/142) - Add NTP time synchronization. Optional dependency - install with `pip install mio[ntp]`
- [`#147`](https://github.com/miniscope/mio/pull/147) - `buffer_npix` is now a property on `StreamDevConfig`

#### Bugfixes
- [`#137`](https://github.com/miniscope/mio/pull/137) - Fix reconstructed frame index error in `mio stream`
- [`#148`](https://github.com/miniscope/mio/pull/148) - Make OpalKelly driver import optional so CLI works without device drivers

#### Maintenance
- [`#149`](https://github.com/miniscope/mio/pull/149) - Pin numpy <2.0 and remove deprecated ruff rules
- Exclude Python 3.13 on Windows from test matrix, fix docs build errors

## 0.8

### 0.8.1 - 2025-09-25 - Bugfix for frame reconstruction, CLI updates, more metadata for `stream`

#### New features
- [`#93`](https://github.com/Aharoni-Lab/mio/pull/93) - Add `mio config list` to display available configs
- [`#103`](https://github.com/Aharoni-Lab/mio/pull/103) - Add `mio hash` to cli to hash video files
- [`#136`](https://github.com/Aharoni-Lab/mio/pull/136) - Add runtime metadata fields for later video repair

#### Bugfixes
- [`#136`](https://github.com/Aharoni-Lab/mio/pull/136) - Fix buffer misalignment during frame reconstruction

### 0.8.0 - 2025-09-19 - Realtime frequency filters for `stream` and test adjustments

#### New Features
- [`#121`](https://github.com/Aharoni-Lab/mio/pull/121) - Adds optional realtime frequency filtering to `mio stream`. Example usage:  
  ```bash
  mio stream capture -c wireless-200px -f remove_stripe_example
  ```  
  See the CLI docs for more details.

#### Bugfixes
- [`#121`](https://github.com/Aharoni-Lab/mio/pull/121) - The FFT/IFFT operation on videos sometimes produced values larger than the maximum of the type (e.g., 255 for `np.uint8`), resulting in bright artifacts. The frequency mask helper method now clips values before returning them to prevent this.

#### Refactors
- [`#121`](https://github.com/Aharoni-Lab/mio/pull/121) - `FrequencyMaskingConfig` is now a subclass of `MiniscopeConfig, ConfigYAMLMixin`, allowing it to be treated like other configs.

#### Testing
- [`#132`](https://github.com/Aharoni-Lab/mio/pull/132), [`#134`](https://github.com/Aharoni-Lab/mio/pull/134) - Pin macOS version to `macos-14` due to upstream issues in the Coveralls action.
- [`#121`](https://github.com/Aharoni-Lab/mio/pull/121) - Add basic tests to ensure the method is invoked via CLI.


## 0.7.*

### 0.7.1 - 2025-08-07

#### Bugfix

- [`#125`](https://github.com/Aharoni-Lab/mio/pull/125) - If a device configuration caused the
  `buffer_npix` method to not match the number of buffers per frame returned by the device,
  frame headers would be duplicated when sent through the rest of the pipeline,
  resulting in duplicate rows in the resulting csv file.
  Behavior was fixed, and the warning was upgrade to a more explicit exception message
  that is more informative about the nature of the problem

#### Added

- [`#125`](https://github.com/Aharoni-Lab/mio/pull/125) - the `BufferedCSVWriter` class now
  takes an explicit list of values to use as the headers, and accepts dicts for rows that
  match those headers. this makes for explicit alignment in produced tables,
  ignoring extra values and filling missing values.
- [`#117`](https://github.com/Aharoni-Lab/mio/pull/117) - A {class}`mio.plots.VideoPlotter`
  class was added to display a list of videos together

#### Refactor

- [`#116`](https://github.com/Aharoni-Lab/mio/pull/116) - Continuing the process of
  splitting up the {class}`mio.stream_daq.StreamDaq` class, the method for
  iterating over an arbitrary-sized input buffer from a device, concatenating it,
  and splitting it into buffers according to some `preamble` bytestring
  was split into a {func}`mio.stream_daq.iter_buffers` generator function,
  where the binary source of devices is now any iterator that yields bytestrings.
  Corresponding `__iter__` methods were added to {class}`mio.devices.opalkelly.okDev`.

#### Testing

- [`#125`](https://github.com/Aharoni-Lab/mio/pull/125) - Added a performance test with a generous
  baseline to ensure that the streamdaq runs without significant delays.

### 0.7.0 - 2025-07-21 - Processing Module & Video Encoding Fixes
PRs: [`#83`](https://github.com/Aharoni-Lab/mio/pull/83), [`#94`](https://github.com/Aharoni-Lab/mio/pull/94), [`#97`](https://github.com/Aharoni-Lab/mio/pull/97), [`#99`](https://github.com/Aharoni-Lab/mio/pull/99), [`#112`](https://github.com/Aharoni-Lab/mio/pull/112)
#### Changes
- **Added video processing (denoising) module for neural recordings.** This module is currently for offline use with video files. Real-time integration with `streamDaq` is planned for a future update.
    - Includes features like **broken frame detection** and **horizontal stripe removal**.
    - Refer to `Preprocessing videos` in the documentation for more details.
- **Resolved video encoding bug in export.** Exported videos are now **deterministic**, as intended for the current uncompressed setup.
    - The `VideoWriter` class was also **isolated from the `streamDaq` module** for broader usability.



## 0.6 - Becoming `mio`

### 0.6.0 - 24-12-10

#### Breaking Changes

- `miniscope-io` is now known as `mio`! Big thanks to [`@heuer`](https://github.com/Aharoni-Lab/mio/issues/77)
  for graciously giving us the name. This gives us a nice, short name that is uniform
  across pypi, the repository, and the cli.
- The {meth}`mio.models.config.LogConfig.level_file` and {meth}`mio.models.config.LogConfig.level_stdout`
  fields are no longer automatically populated from the `level` field. 
  This was because of the way the multi-source config system propagates values between
  sources with different priorities. Now downstream consumers should check if these values
  are `None` and use the `level` field if so. 

#### Config

Two big changes to config:

- [`#72`](https://github.com/Aharoni-Lab/mio/pull/72) - `@sneakers-the-rat` - Global config, user config
  from multiple sources: see the [config](../guide/config.md) documentation for more
- [`#76`](https://github.com/Aharoni-Lab/mio/pull/76) - `@sneakers-the-rat` - Convert `formats` to `yaml`.
  We finally got rid of the godforsaken self-inflicted wound of having instantiated models
  serve as config, and instead are using `yaml` everywhere for static config. This includes
  every yaml-able config having a header that indicates which model the config corresponds to,
  a (locally) unique id, which can be used anywhere a path can be, and a version stamp in anticipation
  of being able to handle model migrations.

#### CI

- [`#75`](https://github.com/Aharoni-Lab/mio/pull/75) - `@sneakers-the-rat` - Test docs builds on PRs
  to avoid broken links and references

## 0.5

### 0.5.0 - 24-11-11
Enhancements and bugfixes to `StreamDaq`; adding `device_update` module; CI/CD updates.

#### Features / bugfixes
- **Over-the-air device config:** modules and commands for updating and rebooting; *e.g.,* `mio update --key LED --value 10`, `mio device --reboot`.
- **Continuous run:** updated error handling to continuously capture even when the data stream is interrupted.
- **UNIX timestamp:** added UNIX timestamp to metadata file export.
- **More Opal Kelly bitfiles:** added FPGA configuration images and organized them based on Manchester encoding conventions, frequency, etc.
#### CI/CD
- Switched to `pdm` from `poetry`; now `pdm install --with all` for contributing.
- Added workflow for readthedocs preview link in PRs.
- Added snake_case enforcement (Lint).

Related PRs: [#45](https://github.com/Aharoni-Lab/mio/pull/45), [#48](https://github.com/Aharoni-Lab/mio/pull/48), [#49](https://github.com/Aharoni-Lab/mio/pull/49), [#50](https://github.com/Aharoni-Lab/miniscope-io/pull/50), [#53](https://github.com/Aharoni-Lab/miniscope-io/pull/53), 
Contributors: [@t-sasatani](https://github.com/t-sasatani), [@sneakers-the-rat](https://github.com/sneakers-the-rat), [@MarcelMB](https://github.com/MarcelMB), [@phildong](https://github.com/phildong)

## 0.4

### 0.4.1 - 24-09-01

#### Added Features
- Support dummy words at the beginning of buffer to make data detection in `StreamDaq` more robust. The length of this word can be set via the device config YAML file.
- `models.stream.ADCScaling` class for converting ADC raw values into voltage. The scaling factors can be set via the device config YAML file.

Related PRs: [#41](https://github.com/Aharoni-Lab/miniscope-io/pull/41), [#44](https://github.com/Aharoni-Lab/miniscope-io/pull/44)
Related Issues: [#36](https://github.com/Aharoni-Lab/miniscope-io/issues/36)
Contributors: [@t-sasatani](https://github.com/t-sasatani), [@sneakers-the-rat](https://github.com/sneakers-the-rat), [@MarcelMB](https://github.com/MarcelMB).

### 0.4.0 - 24-08-27

Enhancements and bugfixes to `StreamDaq`. Mostly around handling metadata.

Contributors: [@t-sasatani](https://github.com/t-sasatani), [@sneakers-the-rat](https://github.com/sneakers-the-rat), [@MarcelMB](https://github.com/MarcelMB).
Related PRs: [#35](https://github.com/Aharoni-Lab/miniscope-io/pull/35), [#38](https://github.com/Aharoni-Lab/miniscope-io/pull/38), [#33](https://github.com/Aharoni-Lab/miniscope-io/pull/33)
Related Issues: [#34](https://github.com/Aharoni-Lab/miniscope-io/issues/34), [#36](https://github.com/Aharoni-Lab/miniscope-io/issues/36), [#38](https://github.com/Aharoni-Lab/miniscope-io/issues/38), [#40](https://github.com/Aharoni-Lab/miniscope-io/issues/40)

**Features:**
- **Metadata handling**: CSV export and real-time plotting of metadata. The plotting feature is experimental and is still very unstable. Also improved metadata error handling.
- **CLI enhancement**: Generate video and metadata csv files with same stem names.

**Models/Formats:**
- **Runtime config**: `StreamDevRuntime` now configures queue and buffer sizes used in `StreamDaq`.

**CI / Bugfix:**
- **Fix termination**: Now buffer size doesn't affect output video. Very good for tests.
- **Device docs section**: For adding documentation about specific devices.
- **CLI docs on RTD**: Mock binaries so CLI docs show up on RTD.
- **Dependency**: Update some packages for Windows compatibility.

## 0.3

### 0.3.0 - 24-07-04

Enhancements to `StreamDaq` and improved CI processes.

Contributors: [@sneakers-the-rat](https://github.com/sneakers-the-rat),  [@t-sasatani](https://github.com/t-sasatani).
Related PRs and Issues: [#29](https://github.com/Aharoni-Lab/miniscope-io/pull/29), [#31](https://github.com/Aharoni-Lab/miniscope-io/pull/31), [#27](https://github.com/Aharoni-Lab/miniscope-io/issues/27)

**Features:**
- **Performance enhancements**: Refactoring of streamDaq to eliminate delays in the default configuration.
- **New CLI**: Introduced a `click`-based CLI offering greater flexibility and scalability.
- **Test coverage**: Integrated Coveralls to display test coverage statistics.
- **BufferFormatter class**: Added for flexible bit/byte operations and decoding incoming buffers.

**Models/Formats:**
- **Configuration changes**: Isolated device config and user config. device config is now stored in `StreamDevConfig` (renamed from `StreamDaqConfig`). User configs are segregated to the CLI.

**CI:**
- **Python 3.12 compatibility**: Added tests for Python 3.12.
- **StreamDaq tests**: Fixed issues with video hash assert tests in `streamDaq`.
- **Global test timeout**: added global timeout for all tests.

## 0.2

### 0.2.1 - 24-06-21

Linting and code formatting :)

Added `black` and `ruff` for linting and formatting, 
reformatted the package.

See the [Contributing](./contributing.md) documentation for 
details and usage.

### 0.2.0 - 24-06-21

StreamDaq enhancements and testing

- https://github.com/Aharoni-Lab/miniscope-io/pull/26

Testing:

- [@t-sasatani](https://github.com/t-sasatani) - add end-to-end test for {class}`~mio.stream_daq.streamDaq`
- Add a mock class for {class}`~mio.devices.opalkelly.okDev`
- replace `tmpdir` fixture and `tempfile` module with `tmp_path`

New:

- [@t-sasatani](https://github.com/t-sasatani) - allow use of okDev on Windows
- {meth}`~mio.stream_daq.StreamDaq.capture` can export video :)
- More specific exceptions:
  - {class}`~mio.exceptions.StreamError`
  - {class}`~mio.exceptions.StreamReadError`
  - {class}`~mio.exceptions.DeviceError`
  - {class}`~mio.exceptions.DeviceOpenError`
  - {class}`~mio.exceptions.DeviceConfigurationError`
- {func}`~mio.utils.hash_video` - hash decoded video frames, rather than encoded video file


Fixed:

- Removed `print` statements in {class}`~mio.devices.opalkelly.okDev`
- {meth}`~mio.stream_daq.StreamDaq.capture`
  - Don't require `config`
  - Replace logging with {func}`~mio.logging.init_logger`
  - Use of {attr}`~mio.stream_daq.StreamDaq.terminate` to control inner loops


Models:

- added `fs` and `show_video` to {class}`~mio.models.stream.StreamDaqConfig`

CI:

- [@t-sasatani](https://github.com/t-sasatani) - restore windows and mac tests (oops)
- caching dependency installs
- not using pytest-emoji, it was always annoying

## 0.1

### 0.1.8 - 24-06-16

- https://github.com/Aharoni-Lab/miniscope-io/pull/21
- https://github.com/Aharoni-Lab/miniscope-io/pull/15

New features:

- **Support for Various Image Formats**: `streamDaq` now supports multiple image formats, including different image sizes, frame rates (FPS), and bit-depths. These configurations can be provided via a YAML file. Examples of these configurations can be found in `mio.data.config`.
- **Pydantic Model for Configuration**: Added a Pydantic model to validate the configuration of `streamDaq`.
- **Bitstream Loader**: Added a bitstream loader to automatically configure the Opal Kelly FPGA when running `streamDaq`.
- **Updated Command Line Script**: The command line script for running `streamDaq` has been updated. Use `streamDaq -c path/to/config/yaml/file.yml` to run the process with your YAML configuration file.
- **Logger Module**: Added a logger module that can be configured using environmental variables or a `.env` file.

Note: Version 0.1.7 was skipped accidentally and does not exist.

### 0.1.6 - 24-04-09

- https://github.com/Aharoni-Lab/miniscope-io/pull/14

New features:

- Added support for the wireless FPGA and UART daqs - work in progress unifying the API, but
  initial version of code is present in `stream_daq.py`
- Vendored opalkelly device drivers - see `devices` and `vendor`

### 0.1.5 - 23-09-03

- https://github.com/Aharoni-Lab/miniscope-io/pull/9
- https://github.com/Aharoni-Lab/miniscope-io/pull/10

Bugfixes:
- Handle absolute paths correctly on windows, which can't deal with {meth}`pathlib.Path.resolve()`, apparently

New features:
- Added {meth}`~mio.io.SDCard.to_video` to export videos
  - Added notebook demonstrating {meth}`~mio.io.SDCard.to_video`
- Added {mod}`mio.utils` module with {func}`~.utils.hash_file` function for hashing files (used in testing)

Code structure:
- (Minor) moved {meth}`~mio.io.SDCard.skip` to general methods block (no change)

Tests:
- Run tests on macos and windows

### 0.1.4 - 23-09-03

https://github.com/Aharoni-Lab/miniscope-io/pull/8

New features:

- Data models! Hold a collection of frames and get their headers
- Plots! Mimic the plots from ye olde notebook
- Update to pydantic v2
- Version field in formats
- Format for miniscope firmware with battery voltage level

Reverted:

- grab_frames notebook is restored to using the example data and having the evaluated output present



### 0.1.1 - 23-07-13

#### Additions

- Added {class}`~mio.exceptions.EndOfRecordingException` when attempting to read past last frame
- Added {attr}`~mio.io.SDCard.frame_count` property inferred from the number of buffers and buffers per frame
- Return `self` when entering {class}`~.SDCard` context
- Optionally return {class}`~mio.sd.DataHeader`s from frame when reading

#### Bugfixes

- Index the position of the 0th frame in {attr}`~.SDCard.positions`
- reset internal frame counter to 0 when exiting context