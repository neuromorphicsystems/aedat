import aedat
import pathlib
dirname = pathlib.Path(__file__).resolve().parent

decoder = aedat.Decoder(dirname / 'test_data.aedat4')

assert len(decoder.id_to_stream().keys()) == 4
assert decoder.id_to_stream()[0]['type'] == 'events'
assert decoder.id_to_stream()[0]['width'] == 346
assert decoder.id_to_stream()[0]['height'] == 260
assert decoder.id_to_stream()[1]['type'] == 'frame'
assert decoder.id_to_stream()[1]['width'] == 346
assert decoder.id_to_stream()[1]['height'] == 260
assert decoder.id_to_stream()[2]['type'] == 'imus'
assert decoder.id_to_stream()[3]['type'] == 'triggers'
counts = [0, 0, 0, 0]
sizes = [0, 0, 0, 0]
for packet in decoder:
    counts[packet['stream_id']] += 1
    if 'events' in packet:
        sizes[0] += packet['events'].nbytes
    elif 'frame' in packet:
        sizes[1] += packet['frame']['pixels'].nbytes
    elif 'imus' in packet:
        sizes[2] += packet['imus'].nbytes
    elif 'triggers' in packet:
        sizes[3] += packet['triggers'].nbytes
assert counts[0] == 236
assert counts[1] == 59
assert counts[2] == 236
assert counts[3] == 177
assert sizes[0] == 1024790
assert sizes[1] == 5307640
assert sizes[2] == 113424
assert sizes[3] == 2124
