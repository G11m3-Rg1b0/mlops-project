import os
import wave
import re
from typing import Tuple, Any
import pylab
from PIL import Image
import numpy as np
import io
from pathlib import Path
from abc import ABCMeta, abstractmethod

import librosa


class FileSysManager:
    """Get the output directory of the prepared file."""

    def build_output_path(self, file_path: str, output_data_dir: Path) -> str:
        """Build the path to save prepared data.

        args:
            file_path: Complete path to audio file.
            output_data_dir: Directory to store prepared data in.

        return:
            The full path to the prepared file without the file extension.
        """
        file_name = Path(file_path).stem
        dir_class = f"class_{self.get_class(file_name)}"

        output_dir = os.path.join(output_data_dir, "audio-images", dir_class)
        os.makedirs(output_dir, exist_ok=True)

        return os.path.join(output_dir, file_name)

    @staticmethod
    def get_class(file_name: str) -> str:
        """Extract the data point class from file name.

        args:
            file_name: File name.

        return:
            The class number of the data point as a string.
        """
        return re.search(r'\d+(?=_)', file_name).group(0)


class DataFormatter(metaclass=ABCMeta):
    """Abstract class to convert audio files into their spectrogram in a particular format."""

    @abstractmethod
    def build_spectrogram(self, file_path: str) -> Any:
        pass

    @abstractmethod
    def save(self, spectrogram: Any, path: str):
        pass


class PngFormatter(DataFormatter):
    """Convert audio files into spectrogram in .png files."""

    def build_spectrogram(self, file_path: str) -> Image:
        """Build the image of the spectrogram.

        args:
            file_path: The path to the audio file.

        return:
            A PIL image of the audio spectrogram.
        """
        sound_info, frame_rate = self._get_wav_info(file_path)

        pylab.specgram(sound_info, Fs=frame_rate)

        buffer = io.BytesIO()
        pylab.savefig(buffer, format='png')
        buffer.seek(0)
        return Image.open(buffer)

    def save(self, img: Image, path: str):
        """save spectrogram to path.

        args:
            img: The audio's spectrogram.
            path: The complete prepared file path without the file extension.
        """
        img.save(path + '.png', format='png')
        img.close()

    @staticmethod
    def _get_wav_info(file_path: str) -> Tuple[pylab.ndarray, int]:
        """Function to get sound and frame rate info.

        args:
            file_path: The path to the audio file.

        return:
            A tuple of the sound info and the frame rate of the audio file.
        """
        with wave.open(file_path, "r") as wav:
            frames = wav.readframes(-1)
            sound_info = pylab.frombuffer(frames, "int16")
            frame_rate = wav.getframerate()
        return sound_info, frame_rate


class ArrFormatter(DataFormatter):
    """Convert audio files into spectrogram as array and save it in .csv files."""

    def build_spectrogram(self, file_path: str) -> np.ndarray:
        """Build the spectrogram of the audio and store it as a 2D array.

        args:
            file_path: The path to the audio file.

        return:
            A numpy array representing the audio spectrogram.
        """
        audio, _ = librosa.load(file_path, sr=None, dtype=np.float32)
        return np.abs(librosa.stft(audio, n_fft=169, hop_length=80))

    def save(self, arr: np.ndarray, path: str):
        """save spectrogram to path.

        args:
            arr: The audio's spectrogram.
            path: The complete prepared file path without the file extension.
        """
        np.save(path + '.csv', arr)
