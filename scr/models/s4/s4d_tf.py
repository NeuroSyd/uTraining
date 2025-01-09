import tensorflow as tf
import tensorflow_complex as tfc
from tensorflow import nn
from tensorflow.keras.layers import Conv1D, Activation, LayerNormalization
import math

class S4DKernel(tf.keras.layers.Layer):
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super(S4DKernel, self).__init__()
        H = d_model
        log_dt = tf.math.log(dt_min) + tf.random.uniform((H,)) * (tf.math.log(dt_max) - tf.math.log(dt_min))

        C = tf.complex(tf.random.normal((H, N // 2)), tf.random.normal((H, N // 2)))
        self.C = self.add_weight("C", shape=(H, N // 2), initializer=tf.keras.initializers.Zeros())
        self.log_dt = self.add_weight("log_dt", shape=(H,), initializer=tf.keras.initializers.Constant(log_dt), trainable=True)
        log_A_real = tf.math.log(0.5) * tf.ones((H, N // 2))
        A_imag = math.pi * tf.repeat(tf.range(N // 2, dtype=tf.float32), H)
        self.log_A_real = self.add_weight("log_A_real", shape=(H, N // 2), initializer=tf.keras.initializers.Constant(log_A_real), trainable=True)
        self.A_imag = self.add_weight("A_imag", shape=(H, N // 2), initializer=tf.keras.initializers.Constant(A_imag), trainable=True)

    def call(self, L):
        dt = tf.math.exp(self.log_dt)
        C = self.C
        A = -tf.math.exp(self.log_A_real) + 1j * self.A_imag

        dtA = A * tf.expand_dims(dt, axis=-1)
        K = dtA[:, tf.newaxis, :] * tf.range(L, dtype=tf.complex64)
        C = C * (tf.math.exp(dtA) - 1.) / A
        K = 2 * tf.reduce_sum(C[:, tf.newaxis, :] * tf.exp(K), axis=-2)

        return tf.math.real(K)

class S4D(tf.keras.Model):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super(S4D, self).__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed
        self.D = self.add_weight("D", shape=(self.h,))

        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)
        self.activation = Activation("gelu")
        self.dropout = tf.keras.layers.Dropout(dropout) if dropout > 0.0 else tf.keras.layers.Identity()
        self.output_linear = tf.keras.Sequential([
            Conv1D(2 * self.h, kernel_size=1),
            nn.GLU(axis=-2)
        ])

    def call(self, u, **kwargs):
        if not self.transposed:
            u = tf.transpose(u, perm=[0, 2, 1])
        L = u.shape[-1]

        k = self.kernel(L)
        k_f = tf.signal.rfft(k, fft_length=2 * L)
        u_f = tf.signal.rfft(u, fft_length=2 * L)
        y = tf.signal.irfft(u_f * k_f, fft_length=2 * L)[:,:,:L]
        y = y + u * tf.expand_dims(self.D, axis=-1)
        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = tf.transpose(y, perm=[0, 2, 1])
        return y, None