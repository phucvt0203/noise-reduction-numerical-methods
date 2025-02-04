import librosa
import time
import numpy as np
import IPython.display as ipd
import numpy as np
import soundfile as sf
import pandas as pd
import argparse

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, FastICA
from numpy.linalg import svd, norm
from multiprocessing.pool import ThreadPool
from scipy.signal import butter, lfilter
from scipy.ndimage import median_filter

import warnings

warnings.filterwarnings('ignore')

import time
from numpy import *
from numpy.linalg import svd, norm
from multiprocessing.pool import ThreadPool

class RPCA_ADMM:
    def __init__(self, data, g2_factor=0.15, g3_factor=0.15, max_iter=100, abstol=1e-4, reltol=1e-2):
        self.data = float_(data)
        self.m, self.n = self.data.shape
        self.g2_max = norm(hstack(self.data).T, inf)
        self.g3_max = norm(self.data, 2)
        self.g2 = g2_factor * self.g2_max
        self.g3 = g3_factor * self.g3_max
        self.max_iter = max_iter
        self.abstol = abstol
        self.reltol = reltol
        self.lambdap = 1.0
        self.rho = 1.0 / self.lambdap

        self.X_1 = zeros((self.m, self.n))
        self.X_2 = zeros((self.m, self.n))
        self.X_3 = zeros((self.m, self.n))
        self.z = zeros((self.m, 3 * self.n))
        self.U = zeros((self.m, self.n))
        
        self.pool = ThreadPool(processes=3)

    @staticmethod
    def prox_l1(v, lambdat):
        """
        Hàm xử lý proximal của norm L1.

        prox_l1(v, lambdat) là hàm xử lý proximal của norm L1
        với tham số lambdat.
        """
        return maximum(0, v - lambdat) - maximum(0, -v - lambdat)

    @staticmethod
    def prox_matrix(v, lambdat, prox_f):
        """
        Hàm xử lý proximal của một hàm ma trận.

        Giả sử F là một hàm ma trận invariant orthogonal sao cho
        F(X) = f(s(X)), trong đó s là ánh xạ giá trị kỳ dị và f là một
        hàm đối xứng tuyệt đối. Sau đó

        X = prox_matrix(V, lambdat, prox_f)

        đánh giá hàm xử lý proximal của F thông qua hàm xử lý proximal
        của f. Ở đây, cần có khả năng đánh giá prox_f như prox_f(v, lambdat).

        Ví dụ,

        prox_matrix(V, lambdat, prox_l1)

        đánh giá hàm xử lý proximal của norm hạt nhân tại V
        (tức là, hàm ngưỡng giá trị kỳ dị).
        """
        U, S, V = svd(v, full_matrices=False)
        S = S.reshape((len(S), 1))
        pf = diagflat(prox_f(S, lambdat))
        return U.dot(pf).dot(V.conj())

    @staticmethod
    def avg(*args):
        """
        Tính trung bình của các đối số truyền vào.
        """
        N = len(args)
        x = 0
        for k in range(N):
            x = x + args[k]
        x = x / N
        return x

    def objective(self, X_1, g_2, X_2, g_3, X_3):
        """
        Hàm mục tiêu cho RPCA:
            Noise - norm Frobenius bình phương (làm cho X_i nhỏ)
            Background - norm hạt nhân (làm cho X_i hạng thấp)
            Foreground - norm L1 từng phần tử (làm cho X_i nhỏ)
        """
        tmp = svd(X_3, compute_uv=0)
        tmp = tmp.reshape((len(tmp), 1))
        return norm(X_1, 'fro')**2 + g_2 * norm(hstack(X_2), 1) + g_3 * norm(tmp, 1)

    def x1update(self, x, b, l):
        return (1.0 / (1.0 + l)) * (x - b)

    def x2update(self, x, b, l, g, pl):
        return pl(x - b, l * g)

    def x3update(self, x, b, l, g, pl, pm):
        return pm(x - b, l * g, pl)

    def update(self, func, item):
        return list(map(func, [item]))[0]

    def run(self):
        start = time.time()
        """
        Triển khai ADMM cho phân rã ma trận. Trong trường hợp này là RPCA.

        Adapted from: http://web.stanford.edu/~boyd/papers/prox_algs/matrix_decomp.html
        """
        h = {}
        h['objval'] = zeros(self.max_iter)
        h['r_norm'] = zeros(self.max_iter)
        h['s_norm'] = zeros(self.max_iter)
        h['eps_pri'] = zeros(self.max_iter)
        h['eps_dual'] = zeros(self.max_iter)

        print('\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s' % ('iter',
                                                      'r norm',
                                                      'eps pri',
                                                      's norm',
                                                      'eps dual',
                                                      'objective'))

        for k in range(self.max_iter):
            B = self.avg(self.X_1, self.X_2, self.X_3) - self.data / 3 + self.U

            # Cập nhật x ban đầu 
            # X_1 = (1.0 / (1.0 + lambdap)) * (X_1 - B)
            # X_2 = prox_l1(X_2 - B, lambdap * g2)
            # X_3 = prox_matrix(X_3 - B, lambdap * g3, prox_l1)

            # Cập nhật x song song
            async_X1 = self.pool.apply_async(self.update, (lambda x: self.x1update(x, B, self.lambdap), self.X_1))
            async_X2 = self.pool.apply_async(self.update, (lambda x: self.x2update(x, B, self.lambdap, self.g2, self.prox_l1), self.X_2))
            async_X3 = self.pool.apply_async(self.update, (lambda x: self.x3update(x, B, self.lambdap, self.g3, self.prox_l1, self.prox_matrix), self.X_3))

            self.X_1 = async_X1.get()
            self.X_2 = async_X2.get()
            self.X_3 = async_X3.get()

            # (chỉ cho kiểm tra kết thúc)
            x = hstack([self.X_1, self.X_2, self.X_3])
            zold = self.z
            self.z = x + tile(-self.avg(self.X_1, self.X_2, self.X_3) + self.data / 3.0, (1, 3))

            self.U = B
            
            # chuẩn đoán, báo cáo, kiểm tra kết thúc
            h['objval'][k] = self.objective(self.X_1, self.g2, self.X_2, self.g3, self.X_3)
            h['r_norm'][k] = norm(x - self.z, 'fro')
            h['s_norm'][k] = norm(-self.rho * (self.z - zold), 'fro')
            h['eps_pri'][k] = sqrt(self.m * self.n * 3) * self.abstol + self.reltol * maximum(norm(x, 'fro'), norm(-self.z, 'fro'))
            h['eps_dual'][k] = sqrt(self.m * self.n * 3) * self.abstol + self.reltol * sqrt(3) * norm(self.rho * self.U, 'fro')

            if (k == 0) or ((k + 1) % 10 == 0):
                print('%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f' % (k + 1,
                                                                       h['r_norm'][k],
                                                                       h['eps_pri'][k],
                                                                       h['s_norm'][k],
                                                                       h['eps_dual'][k],
                                                                       h['objval'][k]))
            if (h['r_norm'][k] < h['eps_pri'][k]) and (h['s_norm'][k] < h['eps_dual'][k]):
                break
            
        h['addm_toc'] = time.time() - start
        h['admm_iter'] = k
        h['X1_admm'] = self.X_1
        h['X2_admm'] = self.X_2
        h['X3_admm'] = self.X_3

        return h

class AudioDenoiser:
    def __init__(self, signal, sample_rate):
        self.signal = signal
        self.sample_rate = sample_rate

    def denoise_audio_with_PCA(self, rate_components=0.9, block_size=1024):
        """
        Khử nhiễu âm thanh bằng PCA.

        Parameters:
        - rate_components: Tỷ lệ thành phần chính được giữ lại (mặc định là 0.9).
        - block_size: Kích thước khối để chia tín hiệu (mặc định là 1024).

        Returns:
        - denoised: Tín hiệu âm thanh đã được khử nhiễu.
        """
        samples = len(self.signal)
        hanging = block_size - np.mod(samples, block_size)
        padded = np.pad(self.signal, (0, hanging), 'constant', constant_values=0)
        reshaped = padded.reshape((-1, block_size))
        
        pca = PCA(rate_components)
        denoised = pca.inverse_transform(pca.fit_transform(reshaped))
        
        denoised = denoised.reshape(-1)[:samples]
        return denoised

    def low_pass_filter(self, cutoff=3000, order=5):
        """
        Áp dụng bộ lọc thông thấp cho dữ liệu âm thanh.

        Parameters:
        - cutoff: Tần số cắt (cutoff frequency) của bộ lọc (mặc định là 3000 Hz).
        - order: Bậc của bộ lọc Butterworth (mặc định là 5).

        Returns:
        - y: Dữ liệu âm thanh đã được lọc, chỉ giữ lại các tần số thấp hơn tần số cắt.
        """
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, self.signal)
        return y

    def spectral_gate_filter(self, stft, n_std_thresh=1.0):
        """
        Áp dụng bộ lọc ngưỡng phổ cho ma trận STFT của tín hiệu âm thanh.

        Parameters:
        - stft: Ma trận STFT của tín hiệu âm thanh, trong đó mỗi cột đại diện cho phổ của một đoạn ngắn của tín hiệu.
        - n_std_thresh: Số lần độ lệch chuẩn để xác định ngưỡng (mặc định là 1.0).

        Returns:
        - filtered_stft: Ma trận STFT đã được lọc, chỉ giữ lại các thành phần phổ có biên độ lớn hơn hoặc bằng ngưỡng xác định.
        """
        mean_spec = np.mean(stft, axis=1, keepdims=True)
        std_spec = np.std(stft, axis=1, keepdims=True)
        threshold = mean_spec + n_std_thresh * std_spec
        mask = stft >= threshold
        filtered_stft = stft * mask
        return filtered_stft

    def denoise_audio_with_RPCA(self, is_spectral_gate_filter=True, is_low_pass_filter=True, n_std_thresh=1.0, cutoff=3000):
        """
        Khử nhiễu âm thanh bằng RPCA với các tùy chọn lọc phổ và lọc thông thấp.

        Parameters:
        - is_spectral_gate_filter: Áp dụng bộ lọc ngưỡng phổ (mặc định là False).
        - is_low_pass_filter: Áp dụng bộ lọc thông thấp (mặc định là False).
        - n_std_thresh: Số lần độ lệch chuẩn để xác định ngưỡng (mặc định là 1.0).
        - cutoff: Tần số cắt (cutoff frequency) của bộ lọc thông thấp (mặc định là 3000 Hz).

        Returns:
        - denoised_samples: Tín hiệu âm thanh đã được khử nhiễu.
        """
        n_fft = 2048
        hop_length = 512
        stft = librosa.stft(self.signal, n_fft=n_fft, hop_length=hop_length)

        magnitude, phase = np.abs(stft), np.angle(stft)

        if is_spectral_gate_filter:
            magnitude = self.spectral_gate_filter(magnitude, n_std_thresh)

        magnitude = median_filter(magnitude, size=(3, 3))

        rpca = RPCA_ADMM(magnitude)
        result = rpca.run()
        L = result['X1_admm']
        S = result['X2_admm']

        denoised_stft = L * np.exp(1j * phase)
        denoised_samples = librosa.istft(denoised_stft, hop_length=hop_length)

        if is_low_pass_filter:
            denoised_samples = self.low_pass_filter(cutoff=cutoff)

        return denoised_samples
    
    def denoise_audio_with_ICA(self, signal2=None):
        if signal2 is None:
            raise ValueError("For ICA denoising, two audio signals are required.")
        
        # Chặt cụt để tương ứng
        min_length = min(len(self.signal), len(signal2))
        self.signal = self.signal[:min_length]
        signal2 = signal2[:min_length]

        # Stack
        X = np.vstack((self.signal, signal2))

        #  ICA
        ica = FastICA(n_components = 2)
        S_estimated = ica.fit_transform(X.T).T
        # Stack signals

        return S_estimated[0], S_estimated[1]

def main(args):
    # Load audio files
    try:
        signal1, sample_rate1 = librosa.load(args.input_file1, sr=None)
        if args.input_file2:
            signal2, sample_rate2 = librosa.load(args.input_file2, sr=None)
        else:
            signal2 = None
    except Exception as e:
        print(f"Error loading audio file(s): {str(e)}")
        return
    
    # Create AudioDenoiser objects
    denoiser1 = AudioDenoiser(signal1, sample_rate1)
    denoiser2 = AudioDenoiser(signal2, sample_rate2) if signal2 is not None else None
    
    # Choose denoising method
    if args.method == 'pca':
        denoised_signal1 = denoiser1.denoise_audio_with_PCA(rate_components=args.rate_components, block_size=args.block_size)
        denoised_signal2 = denoiser2.denoise_audio_with_PCA(rate_components=args.rate_components, block_size=args.block_size) if denoiser2 else None
    elif args.method == 'rpca':
        denoised_signal1 = denoiser1.denoise_audio_with_RPCA(is_spectral_gate_filter=args.spectral_gate_filter, is_low_pass_filter=args.low_pass_filter, n_std_thresh=args.n_std_thresh, cutoff=args.cutoff)
        denoised_signal2 = denoiser2.denoise_audio_with_RPCA(is_spectral_gate_filter=args.spectral_gate_filter, is_low_pass_filter=args.low_pass_filter, n_std_thresh=args.n_std_thresh, cutoff=args.cutoff) if denoiser2 else None
    elif args.method == 'ica':
        if denoiser2:
            denoised_signal1, denoised_signal2 = denoiser1.denoise_audio_with_ICA(signal2=signal2)
        else:
            print("For ICA method, please provide two input files.")
            return
    else:
        print("Unsupported denoising method. Choose either 'pca', 'rpca' or 'ica'.")
        return
    
    # Save denoised audios
    try:
        sf.write(args.output_file1, denoised_signal1, sample_rate1)
        print(f"Denoised audio saved to {args.output_file1}")
        if denoised_signal2 is not None:
            sf.write(args.output_file2, denoised_signal2, sample_rate2)
            print(f"Denoised audio saved to {args.output_file2}")
    except Exception as e:
        print(f"Error saving denoised audios: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio denoising using PCA, RPCA, or ICA")
    parser.add_argument('-i1', '--input_file1', type=str, required=True, help="Input audio file path 1")
    parser.add_argument('-i2', '--input_file2', type=str, required=False, help="Input audio file path 2 (required only for ICA)")
    parser.add_argument('-o1', '--output_file1', type=str, required=True, help="Output denoised audio file path 1")
    parser.add_argument('-o2', '--output_file2', type=str, required=False, help="Output denoised audio file path 2")
    parser.add_argument('-m', '--method', type=str, choices=['pca', 'rpca', 'ica'], default='pca',
                        help="Denoising method: 'pca', 'rpca' or 'ica' (default: pca)")
    
    # Arguments for PCA denoising
    parser.add_argument('--rate_components', type=float, default=0.9,
                        help="Percentage of principal components to keep in PCA (default: 0.9)")
    parser.add_argument('--block_size', type=int, default=1024, help="Block size for PCA denoising (default: 1024)")
    
    # Arguments for RPCA denoising
    parser.add_argument('--spectral_gate_filter', action='store_true', help="Apply spectral gate filter (default: False)")
    parser.add_argument('--low_pass_filter', action='store_true', help="Apply low pass filter (default: False)")
    parser.add_argument('--n_std_thresh', type=float, default=1.0,
                        help="Number of standard deviations for spectral gate filter (default: 1.0)")
    parser.add_argument('--cutoff', type=float, default=3000, help="Cutoff frequency for low pass filter (default: 3000 Hz)")
    
    args = parser.parse_args()
    main(args)