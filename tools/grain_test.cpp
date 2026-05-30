// Headless validation harness for the grain model.
// Renders flat grey patches at several input levels, measures the resulting
// grain mean/stddev, and writes a PGM ramp so the grain can be eyeballed.
//
// Build:  see Makefile target `grain_test`
// Usage:  ./grain_test [grainRadius] [sigmaR] [nSamples] [size]

#include "../src/GrainModel.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

int main(int argc, char** argv) {
    grain::Params p;
    p.grainRadius = argc > 1 ? (float)atof(argv[1]) : 0.85f;
    p.sigmaR      = argc > 2 ? (float)atof(argv[2]) : 0.0f;
    p.nSamples    = argc > 3 ? atoi(argv[3]) : 256;
    int W         = argc > 4 ? atoi(argv[4]) : 256;
    int H         = W;
    p.seed = 2026u;

    printf("grainRadius=%.3f sigmaR=%.3f nSamples=%d patch=%dx%d\n",
           p.grainRadius, p.sigmaR, p.nSamples, W, H);
    printf("\n  input |   mean   |  stddev  | (stddev peaks in mid-tones for real grain)\n");
    printf("  ------+----------+----------\n");

    const float levels[] = {0.05f, 0.10f, 0.18f, 0.30f, 0.50f, 0.70f, 0.90f};
    for (float lvl : levels) {
        double sum = 0.0, sum2 = 0.0;
        int n = W * H;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float v = grain::renderPixel(lvl, x, y, 0, p);
                sum += v; sum2 += (double)v * v;
            }
        double mean = sum / n;
        double var = sum2 / n - mean * mean;
        double sd = var > 0 ? std::sqrt(var) : 0.0;
        printf("  %.2f  |  %.4f  |  %.4f\n", lvl, mean, sd);
    }

    // Write a horizontal grey ramp with grain as a PGM for visual inspection.
    const char* outPath = "grain_ramp.pgm";
    std::vector<unsigned char> img(W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float lvl = (float)x / (float)(W - 1);
            float v = grain::renderPixel(lvl, x, y, 0, p);
            int iv = (int)(v * 255.0f + 0.5f);
            img[y * W + x] = (unsigned char)(iv < 0 ? 0 : (iv > 255 ? 255 : iv));
        }
    FILE* f = fopen(outPath, "wb");
    if (f) {
        fprintf(f, "P5\n%d %d\n255\n", W, H);
        fwrite(img.data(), 1, img.size(), f);
        fclose(f);
        printf("\nWrote %s (%dx%d grey ramp with grain)\n", outPath, W, H);
    }
    return 0;
}
