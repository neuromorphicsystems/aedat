import hashlib
import pathlib

import aedat

dirname = pathlib.Path(__file__).resolve().parent

decoder = aedat.Decoder(dirname / "test_data.aedat4")

assert len(decoder.id_to_stream().keys()) == 4
assert decoder.id_to_stream()[0]["type"] == "events"
assert decoder.id_to_stream()[0]["width"] == 346
assert decoder.id_to_stream()[0]["height"] == 260
assert decoder.id_to_stream()[1]["type"] == "frame"
assert decoder.id_to_stream()[1]["width"] == 346
assert decoder.id_to_stream()[1]["height"] == 260
assert decoder.id_to_stream()[2]["type"] == "imus"
assert decoder.id_to_stream()[3]["type"] == "triggers"
t_hasher = hashlib.sha3_224()
x_hasher = hashlib.sha3_224()
y_hasher = hashlib.sha3_224()
on_hasher = hashlib.sha3_224()
frame_hasher = hashlib.sha3_224()
imus_hasher = hashlib.sha3_224()
triggers_hasher = hashlib.sha3_224()
for packet in decoder:
    if "events" in packet:
        events = packet["events"]
        t_hasher.update(events["t"].tobytes())
        x_hasher.update(events["x"].tobytes())
        y_hasher.update(events["y"].tobytes())
        on_hasher.update(events["on"].tobytes())
    if "frame" in packet:
        frame_hasher.update(packet["frame"]["pixels"].tobytes())
    if "imus" in packet:
        imus_hasher.update(packet["imus"].tobytes())
    if "triggers" in packet:
        triggers_hasher.update(packet["triggers"].tobytes())
print(f"{t_hasher.hexdigest()=}")
print(f"{x_hasher.hexdigest()=}")
print(f"{y_hasher.hexdigest()=}")
print(f"{on_hasher.hexdigest()=}")
print(f"{frame_hasher.hexdigest()=}")
print(f"{imus_hasher.hexdigest()=}")
print(f"{triggers_hasher.hexdigest()=}")
