import os
from bs4 import BeautifulSoup
from PIL import Image, ExifTags
from pymap3d import ecef2enu, geodetic2ecef
import numpy as np


def dms_to_decimal(d, m, s):
    return d + (m / 60.0) + (s / 3600.0)


def get_gps_coords(im):
    """
    Gets latitude and longitude values from image EXIF data.
    :param im:
    :return:
    """
    exif = im.getexif()
    exif_data = dict()
    for tag, value in exif.items():
        decoded_tag = ExifTags.TAGS.get(tag, tag)
        exif_data[decoded_tag] = value
    gps_info = exif_data['GPSInfo']
    lat_dms = map(lambda x: x[0] / float(x[1]), gps_info[2])
    lat = dms_to_decimal(*lat_dms)
    if gps_info[1] == 'S':
        lat *= -1
    lng_dms = map(lambda x: x[0] / float(x[1]), gps_info[4])
    lng = dms_to_decimal(*lng_dms)
    if gps_info[3] == 'W':
        lng *= -1
    return lat, lng


def get_data(path):
    lat0 = None
    lon0 = None
    h0 = 0
    for root, dirs, files in os.walk(path):
        for filename in sorted(filter(lambda x: os.path.splitext(x)[1].lower() == '.jpg', files)):
            filepath = os.path.join(root, filename)
            with Image.open(filepath) as im:
                for segment, content in im.applist:
                    marker, body = content.split('\x00', 1)
                    if segment == 'APP1' and marker == 'http://ns.adobe.com/xap/1.0/':
                        soup = BeautifulSoup(body, features='html.parser')
                        description = soup.find('x:xmpmeta').find('rdf:rdf').find('rdf:description')
                        pitch = float(description['drone-dji:gimbalpitchdegree']) + 90
                        yaw = float(description['drone-dji:gimbalyawdegree'])
                        roll = float(description['drone-dji:gimbalrolldegree'])
                        alt = float(description['drone-dji:relativealtitude'])
                        lat, lon = get_gps_coords(im)
                        if lat0 is None:
                            lat0 = lat
                            lon0 = lon
                        x, y, z = geodetic2ecef(lat, lon, alt)
                        x, y, z = ecef2enu(x, y, z, lat0, lon0, h0)
                        yield filename, '{:f}'.format(x), '{:f}'.format(y), '{:f}'.format(z), yaw, pitch, roll


def main():
    data = [d for d in get_data('datasets/images')]
    data = sorted(data, key=lambda x: x[0])
    x = np.array(map(lambda d: d[1], data))
    y = np.array(map(lambda d: d[2], data))
    with open('datasets/imageData.txt', 'w+') as f:
        for datum in data:
            f.write(','.join([str(d) for d in datum]) + '\n')


if __name__ == '__main__':
    main()
