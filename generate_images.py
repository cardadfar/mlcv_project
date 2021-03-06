import numpy as np
import cairocffi as cairo
import os
import struct
from struct import unpack
import cv2
from multiprocessing import Pool


def vector_to_raster(vector_images, side=28, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1), name='default'):
    """
    padding and line_diameter are relative to the original 256x256 image.
    https://github.com/googlecreativelab/quickdraw-dataset/issues/19#issuecomment-402247262
    """
    
    original_side = 256.
    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    for i, vector_image in enumerate(vector_images):
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()
        
        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1,1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)        
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_image = raster_image.reshape([side, side])
        cv2.imwrite('data/png/{}/{}.png'.format(name, i), raster_image)
    
    return np.array(raster_images)


def process_drawings(path, name):
    """
    https://github.com/googlecreativelab/quickdraw-dataset/blob/master/examples/binary_file_parser.py
    """
    def unpack_drawing(file_handle):
        key_id, = unpack('Q', file_handle.read(8))
        country_code, = unpack('2s', file_handle.read(2))
        recognized, = unpack('b', file_handle.read(1))
        timestamp, = unpack('I', file_handle.read(4))
        n_strokes, = unpack('H', file_handle.read(2))
        image = []
        for i in range(n_strokes):
            n_points, = unpack('H', file_handle.read(2))
            fmt = str(n_points) + 'B'
            x = unpack(fmt, file_handle.read(n_points))
            y = unpack(fmt, file_handle.read(n_points))
            image.append((x, y))

        return {
            'key_id': key_id,
            'country_code': country_code,
            'recognized': recognized,
            'timestamp': timestamp,
            'image': image
        }


    def unpack_drawings(filename):
        with open(filename, 'rb') as f:
            while True:
                try:
                    yield unpack_drawing(f)
                except struct.error:
                    break


    drawings = unpack_drawings(path)
    drawings = list(map(lambda x: x['image'], drawings))
    vector_to_raster(drawings, side=256, name=name)


def worker(filename):
    print('started', filename)
    os.makedirs('data/png/' + filename[:-4], exist_ok=True)
    process_drawings('data/binary/' + filename, filename[:-4])
    print('finished', filename)


if __name__ == '__main__':
    os.makedirs('data/png/', exist_ok=True)
    filenames = os.listdir('data/binary/')
    # with Pool(6) as p:
    #     p.map(worker, filenames)
    for filename in filenames:
        worker(filename)
