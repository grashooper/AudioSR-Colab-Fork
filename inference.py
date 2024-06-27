import gc
import os
import random
import numpy as np
from scipy.signal.windows import hann
import soundfile as sf
import torch
from cog import BasePredictor, Input, Path
import tempfile
import argparse
import librosa
from audiosr import build_model, super_resolution
from scipy import signal
import pyloudnorm as pyln


import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")



def match_array_shapes(array_1:np.ndarray, array_2:np.ndarray):
    if (len(array_1.shape) == 1) & (len(array_2.shape) == 1):
        if array_1.shape[0] > array_2.shape[0]:
            array_1 = array_1[:array_2.shape[0]]
        elif array_1.shape[0] < array_2.shape[0]:
            array_1 = np.pad(array_1, ((array_2.shape[0] - array_1.shape[0], 0)), 'constant', constant_values=0)
    else:
        if array_1.shape[1] > array_2.shape[1]:
            array_1 = array_1[:,:array_2.shape[1]]
        elif array_1.shape[1] < array_2.shape[1]:
            padding = array_2.shape[1] - array_1.shape[1]
            array_1 = np.pad(array_1, ((0,0), (0,padding)), 'constant', constant_values=0)
    return array_1


def lr_filter(audio, cutoff, filter_type, order=12, sr=48000):
    audio = audio.T
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order//2, normal_cutoff, btype=filter_type, analog=False)
    sos = signal.tf2sos(b, a)
    filtered_audio = signal.sosfiltfilt(sos, audio)
    return filtered_audio.T



class Predictor(BasePredictor):
    def setup(self, model_name="basic", device="auto"):
        self.model_name = model_name
        self.device = device
        self.sr = 48000
        print("Loading Model...")
        self.audiosr = build_model(model_name=self.model_name, device=self.device)
        print("Model loaded !")

    def process_audio(self, input_file, chunk_size=5.12, overlap=0.1, seed=None, guidance_scale=3.5, ddim_steps=50):

        audio, sr = librosa.load(input_file, sr=input_cutoff*2, mono=False)
        audio = audio.T
        #print(sr)
        sr = input_cutoff*2
        #sf.write("resampled.wav", audio, sr)
        
        print(f"input cutoff = {input_cutoff}")
        
        # check if audio is stereo & split channels
        is_stereo = len(audio.shape) == 2
        if is_stereo:
            print("audio is stereo")
            audio_ch1, audio_ch2 = audio[:, 0], audio[:, 1]
        else:
            print("Audio is mono")
            audio_ch1 = audio


        # define chunk and overlap size in samples based on input sample rate
        chunk_samples = int(chunk_size * sr)
        #print(chunk_samples)
        overlap_samples = int(overlap * chunk_samples)

        # calculate chunk size and overlap based on output sample rate
        output_chunk_samples = int(chunk_size * self.sr)
        #print(output_chunk_samples)
        output_overlap_samples = int(overlap * output_chunk_samples)
        enable_overlap = True if overlap > 0 else False

        print(f"enable_overlap = {enable_overlap}")
        def process_chunks(audio):
            chunks = []
            original_lengths = []
            start = 0
            while start < len(audio):
                end = min(start + chunk_samples, len(audio))
                chunk = audio[start:end]
                if len(chunk) < chunk_samples:
                    original_lengths.append(len(chunk))
                    pad = np.zeros(chunk_samples - len(chunk))
                    chunk = np.concatenate([chunk, pad])
                else:
                    original_lengths.append(chunk_samples)
                chunks.append(chunk)
                start += chunk_samples - overlap_samples if enable_overlap else chunk_samples
            return chunks, original_lengths

        # create chunks lists for each channel
        chunks_ch1, original_lengths_ch1 = process_chunks(audio_ch1)
        if is_stereo:
            chunks_ch2, original_lengths_ch2 = process_chunks(audio_ch2)

        # process each chunk with the model and reconstruct the audio
        sample_rate_ratio = self.sr / sr
        total_length = len(chunks_ch1) * output_chunk_samples - (len(chunks_ch1) - 1) * (output_overlap_samples if enable_overlap else 0)
        reconstructed_ch1 = np.zeros((1, total_length))
        # print(reconstructed_ch1.shape)

        meter_before = pyln.Meter(sr) # create BS.1770 meter
        meter_after = pyln.Meter(self.sr) # create BS.1770 meter
        

        for i, chunk in enumerate(chunks_ch1):
            loudness_before = meter_before.integrated_loudness(chunk)
            #print(chunk.shape)
            print(f"Processing chunk {i+1} of {len(chunks_ch1)} for Left/Mono channel")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                sf.write(temp_wav.name, chunk, sr)
                # print(f"chunk.shape = {chunk.shape}")

                out_chunk = super_resolution(
                    self.audiosr,
                    temp_wav.name,
                    seed=seed,
                    guidance_scale=guidance_scale,
                    ddim_steps=ddim_steps,
                    latent_t_per_second=12.8
                )
                # remove all junk added by audiosr
                # print(f"out_chunk.shape = {out_chunk.shape}")
                #print(f"1 {out_chunk.shape}")
                out_chunk = out_chunk[0]
                #print(f"2 {out_chunk.shape}")
                # print(f"reshaped out_chunk.shape = {out_chunk.shape}")
                num_samples_to_keep = int(original_lengths_ch1[i] * sample_rate_ratio)
                # print(f"num_samples_to_keep : {num_samples_to_keep}")
                out_chunk = out_chunk[:, :num_samples_to_keep].squeeze()
                #print(f"3 {out_chunk.shape}")

                loudness_after = meter_after.integrated_loudness(out_chunk)
                out_chunk = pyln.normalize.loudness(out_chunk, loudness_after, loudness_before)
                #loudness_after = meter_after.integrated_loudness(out_chunk)
                #print(f"loudness_before = {loudness_before} | loudness_after = {loudness_after}")


                # apply crossfade if overlap is enabled
                if enable_overlap:
                    # calculate the actual overlap size for this chunk
                    actual_overlap_samples = min(output_overlap_samples, num_samples_to_keep)

                    # create fade-out and fade-in arrays of the correct size
                    fade_out = np.linspace(1., 0., actual_overlap_samples)
                    fade_in = np.linspace(0., 1., actual_overlap_samples)

                    if i == 0:
                        out_chunk[-actual_overlap_samples:] *= fade_out

                    elif i < len(chunks_ch1) - 1:
                        out_chunk[:actual_overlap_samples] *= fade_in
                        out_chunk[-actual_overlap_samples:] *= fade_out

                    else:
                        out_chunk[:actual_overlap_samples] *= fade_in

                # print(f"out_chunk.shape : {out_chunk.shape}")

                start = i * (output_chunk_samples - output_overlap_samples if enable_overlap else output_chunk_samples)
                end = start + out_chunk.shape[0]
                reconstructed_ch1[0, start:end] += out_chunk.flatten()


        if is_stereo:
            reconstructed_ch2 = np.zeros((1, total_length))
            # print(reconstructed_ch2.shape)

            for i, chunk in enumerate(chunks_ch2):
                print(f"Processing chunk {i+1} of {len(chunks_ch2)} for Right channel")
                loudness_before = meter_before.integrated_loudness(chunk)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                    sf.write(temp_wav.name, chunk, sr)
                    # print(f"chunk.shape = {chunk.shape}")

                    out_chunk = super_resolution(
                        self.audiosr,
                        temp_wav.name,
                        seed=seed,
                        guidance_scale=guidance_scale,
                        ddim_steps=ddim_steps,
                        latent_t_per_second=12.8
                    )

                    # remove all junk added by audiosr
                    # print(f"out_chunk.shape = {out_chunk.shape}")
                    out_chunk = out_chunk[0]
                    # print(f"reshaped out_chunk.shape = {out_chunk.shape}")
                    num_samples_to_keep = int(original_lengths_ch2[i] * sample_rate_ratio)
                    # print(f"num_samples_to_keep : {num_samples_to_keep}")
                    
                    out_chunk = out_chunk[:, :num_samples_to_keep].squeeze()

                    loudness_after = meter_after.integrated_loudness(out_chunk)
                    out_chunk = pyln.normalize.loudness(out_chunk, loudness_after, loudness_before)
                    #loudness_after = meter_after.integrated_loudness(out_chunk)
                    #print(f"loudness_before = {loudness_before} | loudness_after = {loudness_after}")

                    # apply crossfade if overlap is enabled
                    if enable_overlap:
                        # calculate the actual overlap size for this chunk
                        actual_overlap_samples = min(output_overlap_samples, num_samples_to_keep)

                        # create fade-out and fade-in arrays of the correct size
                        fade_out = np.linspace(1., 0., actual_overlap_samples)
                        fade_in = np.linspace(0., 1., actual_overlap_samples)

                        # no fadein for 1st chunk
                        if i == 0:
                            out_chunk[-actual_overlap_samples:] *= fade_out

                        elif i < len(chunks_ch1) - 1:
                            out_chunk[:actual_overlap_samples] *= fade_in
                            out_chunk[-actual_overlap_samples:] *= fade_out

                        # no fadeout for last  chunk
                        else:
                            out_chunk[:actual_overlap_samples] *= fade_in

                    # print(f"out_chunk.shape : {out_chunk.shape}")

                    start = i * (output_chunk_samples - output_overlap_samples if enable_overlap else output_chunk_samples)
                    end = start + out_chunk.shape[0]
                    reconstructed_ch2[0, start:end] += out_chunk.flatten()

                reconstructed_audio = np.stack([reconstructed_ch1, reconstructed_ch2], axis=-1)
        else:
            reconstructed_audio = reconstructed_ch1


        #print(reconstructed_audio.shape)
        if multiband_ensemble is True:
            # get low from origin input resampled
            #low = librosa.resample(audio.T ,orig_sr=sr,target_sr=48000, res_type='soxr_hq')
            low, _ = librosa.load(input_file, sr=48000, mono=False)
            # print(f"low.shape={low.shape}")

            # fix length issues
            #low = match_array_shapes(low, output)
            output = match_array_shapes(reconstructed_audio[0].T, low)
            #output = match_array_shapes(low, reconstructed_audio[0].T)
            # print(f"low.shape={low.shape}")

            # linkwitz riley crossover
            low = lr_filter(low.T, crossover_freq, 'lowpass', order=10)
            high = lr_filter(output.T, crossover_freq, 'highpass', order=10)
            
            # add smoothing filter to high frequencies (more realistic)
            high = lr_filter(high, 23000, 'lowpass', order=2)

            # print(f"high.shape={high.shape}")

            #sf.write(f"{output_folder}/low_upsampled.wav", low, 48000, subtype='PCM_16')
            #sf.write(f"{output_folder}/high_upsampled.wav", high, 48000, subtype='PCM_16')
            #sf.write(f"{output_folder}/high_full.wav", output.T, 48000, subtype='PCM_16')
            #sf.write(f"{output_folder}/input.wav", audio, 44100, subtype='PCM_16')

            # multiband ensemble
            output = low + high
        
        else:
            output = reconstructed_audio[0]
        # print(f"reconstructed_audio shape : {reconstructed_audio.shape}")
        return output

    def predict(self,
        input_file: Path = Input(description="Audio to upsample"),
        ddim_steps: int = Input(description="Number of inference steps", default=50, ge=10, le=500),
        guidance_scale: float = Input(description="Scale for classifier free guidance", default=3.5, ge=1.0, le=20.0),
        overlap: float = Input(description="overlap size", default=0.04),
        chunk_size: float = Input(description="chunksize", default=10.24),
        seed: int = Input(description="Random seed. Leave blank to randomize the seed", default=None)
    ) -> Path:

        if seed == 0:
            seed = random.randint(0, 2**32 - 1)
        print(f"Setting seed to: {seed}")
        print(f"overlap = {overlap}")
        print(f"guidance_scale = {guidance_scale}")
        print(f"ddim_steps = {ddim_steps}")
        print(f"chunk_size = {chunk_size}")
        print(f"multiband_ensemble = {multiband_ensemble}")
        print(f"input file = {os.path.basename(input_file)}")
        os.makedirs(output_folder, exist_ok=True)
        waveform = self.process_audio(
            input_file,
            chunk_size=chunk_size,
            overlap=overlap,
            seed=seed,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps
        )





        
        filename = os.path.splitext(os.path.basename(input_file))[0]
        sf.write(f"{output_folder}/SR_{filename}.wav", data=waveform, samplerate=48000,  subtype="PCM_16")
        print(f"file created: {output_folder}/SR_{filename}.wav")
        del self.audiosr, waveform
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    """
    ORIGNAL COLAB PARAMETERS FORM UI
    #@markdown #Inference
    input_file_path = '/content/drive/MyDrive/input_file' #@param {type:"string"}
    output_folder = '/content/drive/MyDrive/output_folder' #@param {type:"string"}
    #@markdown ---
    ddim_steps= 20 #@param {type:"slider", min:20, max:200, step:10}
    overlap = 0.04 #@param {type:"slider", min:0, max:0.96, step:0.04}
    guidance_scale=3.5 #@param {type:"slider", min:1, max:15, step:0.5}
    seed = 0 # @param {type:"integer"}
    chunk_size = 10.24 # @param ["5.12", "10.24"] {type:"raw"}
    multiband_ensemble = True # @param {type:"boolean"}
    input_cutoff = "14000" #@param [20000, 19000, 18000, 17000, 16000, 14000, 13000, 13000, 12000, 11000, 10000, 9000, 8000, 7000, 6000, 5000]
    input_cutoff = int(input_cutoff)
    """

    parser = argparse.ArgumentParser(description="Find volume difference of two audio files.")
    parser.add_argument("--input", help="Path to input audio file")
    parser.add_argument("--output", help="Output folder")
    parser.add_argument("--ddim_steps", help="Number of ddim steps", type=int, required=False, default=50)
    parser.add_argument("--chunk_size", help="chunk size", type=float, required=False, default=10.24)
    parser.add_argument("--guidance_scale", help="Guidance scale value",  type=float, required=False, default=3.5)
    parser.add_argument("--seed", help="Seed value, 0 = random seed", type=int, required=False, default=0)
    parser.add_argument("--overlap", help="overlap value", type=float, required=False, default=0.04)
    parser.add_argument("--multiband_ensemble", type=bool, help="Use multiband ensemble with input")
    parser.add_argument("--input_cutoff", help="Define the crossover of audio input in the multiband ensemble", type=int, required=False, default=12000)

    args = parser.parse_args()

    input_file_path = args.input
    output_folder = args.output
    ddim_steps = args.ddim_steps
    chunk_size = args.chunk_size
    guidance_scale = args.guidance_scale
    seed = args.seed
    overlap = args.overlap
    input_cutoff = args.input_cutoff
    multiband_ensemble = args.multiband_ensemble

    crossover_freq = input_cutoff - 1000

    p = Predictor()
    p.setup()
    out = p.predict(
        input_file_path,
        ddim_steps=ddim_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        chunk_size=chunk_size,
        overlap=overlap
    )

    del p
    gc.collect()
    torch.cuda.empty_cache()
