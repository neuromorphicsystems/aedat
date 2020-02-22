import aedat

decoder = aedat.Decoder('/path/to/file.aedat')
print(decoder.id_to_stream())

for packet in decoder:
    print(packet['stream_id'], end=': ')
    if 'events' in packet:
        print('{} polarity events'.format(len(packet['events'])))
    elif 'frame' in packet:
        print('{} x {} frame'.format(packet['frame']['width'], packet['frame']['height']))
    elif 'imus' in packet:
        print('{} IMU samples'.format(len(packet['imus'])))
    elif 'triggers' in packet:
        print('{} trigger events'.format(len(packet['triggers'])))