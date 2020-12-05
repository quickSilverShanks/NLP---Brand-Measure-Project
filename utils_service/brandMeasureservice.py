
from pathlib import Path
from utils_preprocessing.SingalPreprocessing import *
from utils_labelling.SignalLabelsPrediction import signalLabelPrediction
from utils_labelling.SignalLabelling import *
from utils_finaloutputsignal.StoreDiarizedOutput import StoreDiarizedOutput


class BrandMeasureService:

    def performSpeakerDiarization(self, audio_file_path):
        wav_fpath = Path(audio_file_path)
        cont_embeds, wav_splits =process(wav_fpath)
        labels=signalLabelPrediction(cont_embeds)
        labelling = create_labelling(labels,wav_splits)
        output=StoreDiarizedOutput(labelling)


