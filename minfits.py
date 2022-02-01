"""
A utility to write a small (about 37KB) but legal FITS file.  You may
specify the value for the pixels in the 64x64 image; it defaults to 0.
"""
import sys
import astropy.io.fits
import numpy


def min_fits(filepath: str, value: float) -> None:
    image = numpy.full([64, 64], value)
    hdu = astropy.io.fits.PrimaryHDU(image)
    hdu.writeto(filepath)


if __name__ == "__main__":
    args_len = len(sys.argv)
    if args_len not in [2, 3]:
        print("usage: python3 minfits.py <filepath> [<value>]", file=sys.stderr)
        sys.exit(1)

    if args_len == 3:
        _, filepath, value_str = sys.argv
        value = float(value_str)
    elif args_len == 2:
        _, filepath = sys.argv
        value = 0
    else:
        print("The impossible happened.", file=sys.stderr)
        sys.exit(1)

    min_fits(filepath, value)
