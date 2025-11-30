# Installation

From PyPI:

```bash
pip install mio
```

From git repository, using pip:
```bash
git clone https://github.com/Aharoni-Lab/mio
cd mio
pip install .
```

Or pdm:
```bash
git clone https://github.com/Aharoni-Lab/mio
cd mio
pdm install
```


## Additional Dependencies

### OpalKelly

`mio.vendor.opalkelly` - used for FPGA I/O

#### Linux

We package the OpalKelly FrontPanel SDK here, but it has an unadvertised dependency
on some system level packages:

- `liblua5.3-0`

So eg. on debian/ubuntu you'll need to:

```bash
apt install liblua5.3-0
```

#### Mac

No special installation should be required.

#### Windows

Windows Time Service (`w32tm`) may not provide sufficient accuracy for NTP sync checks. For NTP sync check support, install an NTP client. For example, [Meinberg NTP](https://www.meinbergglobal.com/english/sw/ntp.htm) worked well in our setup.

For the `stream` command, you'll need to manually install the [OpalKelly FrontPanel SDK](https://www.opalkelly.com/products/frontpanel/).