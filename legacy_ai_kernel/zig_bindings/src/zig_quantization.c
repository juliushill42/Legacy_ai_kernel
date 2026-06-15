#include <stdint.h>
#include <string.h>

void quantize_int4(const float* input, uint8_t* output, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        uint8_t packed = 0;
        for (size_t j = 0; j < 8 && (i + j) < n; j++) {
            float val = input[i + j];
            int8_t quant = (int8_t)((val + 3.5) * (7.0 / 6.0));
            quant = (quant < -8) ? -8 : (quant > 7) ? 7 : quant;
            packed |= ((quant & 0xF) << (j * 4));
        }
        output[i / 8] = packed;
    }
}

void dequantize_int4(const uint8_t* input, float* output, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint8_t packed = input[i / 8];
        uint8_t bits = (packed >> ((i % 8) * 4)) & 0xF;
        float dequant = (bits - 7) * (6.0 / 7.0) - 3.5;
        output[i] = dequant;
    }
}

void matmul_legacy(const float* a, const float* b, float* c, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}
