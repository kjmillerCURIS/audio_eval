import soundfile as sf
import numpy as np
from collections import namedtuple

_wave_params = namedtuple('_wave_params',
    'nchannels sampwidth framerate nframes comptype compname'
)

class WaveLikeFromSoundFile:
    """Adapter exposing a subset of wave.Wave_read API, backed by soundfile."""
    def __init__(self, path, dtype="int16"):
        self._sf = sf.SoundFile(path, mode="r")
        self._dtype = dtype
        self._sampwidth = np.dtype(dtype).itemsize  # e.g., 2 for int16

    def getnchannels(self):   return self._sf.channels
    def getsampwidth(self):   return self._sampwidth
    def getframerate(self):   return self._sf.samplerate
    def getnframes(self):     return len(self._sf)
    def tell(self):           return self._sf.tell()
    def rewind(self):         self._sf.seek(0)

    def readframes(self, n):
        frames = self._sf.read(frames=n, dtype=self._dtype, always_2d=False)
        return frames.tobytes()

    def getparams(self):
        return _wave_params(
            self.getnchannels(),
            self.getsampwidth(),
            self.getframerate(),
            self.getnframes(),
            'NONE',        # wave module always uses these for PCM
            'not compressed'
        )

    def close(self): self._sf.close()
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.close()

