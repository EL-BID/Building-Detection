import click
import rasterio
import numpy as np

from rastervision.core import Box


@click.command()
@click.argument('min_val')
@click.argument('in_path')
@click.argument('out_path')
def nodata_transform(min_val, in_path, out_path):
    """Convert values less than min_val to 0/NODATA.

    This script was developed to deal with imagery over Belize City which contains values
    that appear to be NODATA values on the periphery of the image, but are actually very
    small values < 7. This script converts those small values to zeros which Raster
    Vision knows to ignore when making predictions.
    """
    min_val = float(min_val)
    chip_size = 1000
    with rasterio.open(in_path, 'r') as in_data:
        with rasterio.open(out_path, 'w', **in_data.profile) as out_data:
            extent = Box(0, 0, in_data.height, in_data.width)
            windows = extent.get_windows(chip_size, chip_size)
            for w in windows:
                # Avoid going off edge of array.
                if w.ymax > in_data.height:
                    w.ymax = in_data.height
                if w.xmax > in_data.width:
                    w.xmax = in_data.width

                print('.', end='', flush=True)
                w = w.rasterio_format()
                im = in_data.read(window=w)

                nodata_mask = np.all(im < min_val, axis=0)

                for b in range(im.shape[0]):
                    band = im[b, :, :]
                    band[nodata_mask] = 0
                    out_data.write_band(b + 1, band, window=w)


if __name__ == '__main__':
    nodata_transform()
