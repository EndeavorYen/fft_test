import numpy as np
import matplotlib.pyplot as plt

def constant_padding(y, m, n, pad_value=0):
    """Constant padding with a specified value."""
    y[m:] = pad_value
    return y

def mirror_padding(y, m, n):
    # Padding method - mirror padding
    # Calculate the number of times the array needs to be repeated
    repeat_times = (n - m) // m + 1
    # Repeat the array and trim to the required size
    mirror_padded = np.tile(y[:m][::-1], repeat_times)[:n - m]
    # Assign to y[m:]
    y[m:] = mirror_padded
    return y

def edge_padding(y, m, n):
    """Edge padding using the edge values of the array."""
    y[m:] = y[m-1]
    return y

def reflect_padding(y, m, n):
    """Reflect padding without repeating the edge value."""
    pad_size = n - m
    y[m:] = np.pad(y[:m], (0, pad_size), mode='reflect', reflect_type='odd')[m:]
    return y

def wrap_padding(y, m, n):
    """Wrap padding using the array values."""
    pad_size = n - m
    y[m:] = np.pad(y[:m], (0, pad_size), mode='wrap')[m:]
    return y

# Parameters
m  = 256         # Number of Sampling Data from Source Signal
n  = 4096         # Total Number of Sampling Data
k1 = 100          # Frequency of source signal
k2 = 105          # Frequency of source signal
s = 1000          # Sampling Rate (Hz)
T = 1.0 / s       # Sampling Time

# ------------------------------------------------------------ #

if __name__ == '__main__':

    x = np.linspace(0.0, n*T, n, endpoint=False)

    # # Create source signal
    low_freq = np.sin(2.0 * 2 * np.pi * x)
    y1 = np.sin(2.0 * k1 * np.pi * x)
    y2 = np.sin(2.0 * k2 * np.pi * x)
    high_freq = np.sin(2.0 * 5000 * np.pi * x)
    y = y1 + y2 + low_freq + high_freq

    # Create pulse signal
    # y = np.zeros(n)
    # y[m//3:] = 1.0

    # Padding method
    # 1. zero padding
    y = constant_padding(y, m, n, pad_value=0)

    # 2. mirror padding
    # y = mirror_padding(y, m, n)

    # 3. edge padding
    # y = edge_padding(y, m, n)

    # 4. reflect padding
    # y = reflect_padding(y, m, n)

    # 5. wrap padding
    # y = wrap_padding(y, m, n)



    # Execute FFT
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(n, T)[:n//2]

    # Calculate magnitude
    magnitude = 2.0/n * np.abs(yf[0:n//2])

    # Plot
    plt.figure(figsize=(12, 6))

    # plot source signal
    plt.subplot(2, 1, 1)  # 2 行 1 列的第 1 個
    plt.plot(x, y)
    plt.title('Original Signal: sin(kx)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()

    # plot FFT result
    plt.subplot(2, 1, 2) # 2rows 1column of 2nd subplot
    plt.plot(xf, magnitude, marker='.', linestyle='-')
    plt.title('FFT of sin(kx)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()

    # added label for peak

    sorted_magn = sorted(magnitude, reverse=True)
    for i in range(len(magnitude)):
        if magnitude[i] > sorted_magn[3]:
            plt.text(xf[i], magnitude[i], f"{xf[i]:.2f}", ha='center')

    # show plot
    plt.tight_layout()  # adjust subplot to fit into figure area
    plt.show()

