import pvporcupine
import pyaudio
import struct
import os

WAKE_TRIGGER_FILE = "wake_trigger.txt"
ACCESS_KEY = "YOUR_PICOVOICE_ACCESS_KEY"  # Replace with your real access key

porcupine = pvporcupine.create(access_key=ACCESS_KEY, keywords=["jarvis"])
pa = pyaudio.PyAudio()
stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

print("Listening for 'jarvis' wake word...")

try:
    while True:
        pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)

        if porcupine.process(pcm_unpacked):
            print("Wake word detected!")

            # Safely write and close the file
            with open(WAKE_TRIGGER_FILE, "w") as f:
                f.write("wake")

            break  # Optionally exit the loop here

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    porcupine.delete()
