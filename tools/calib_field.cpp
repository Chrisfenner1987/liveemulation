// calib_field.cpp - render a flat-field grain coverage image from the model and
// dump raw float32 so numpy can measure its RMS and autocorrelation grain size.
// Used to solve for the grain radius / amount that match the scan.
//
// Usage: ./calib_field <radius> <sigmaR> <nSamples> <level> <size> <out.bin>
#include "../src/GrainModel.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: calib_field radius sigmaR nSamples level size out.bin\n"); return 2; }
    grain::Params p;
    p.grainRadius = atof(argv[1]);
    p.sigmaR      = atof(argv[2]);
    p.nSamples    = atoi(argv[3]);
    float level   = atof(argv[4]);
    int   sz      = atoi(argv[5]);
    const char* out = argv[6];
    p.seed = 2026u;

    std::vector<float> img(sz * sz);
    for (int y = 0; y < sz; ++y)
        for (int x = 0; x < sz; ++x)
            img[y * sz + x] = grain::renderPixel(level, x, y, 0, p);

    FILE* f = fopen(out, "wb");
    fwrite(img.data(), sizeof(float), img.size(), f);
    fclose(f);
    fprintf(stderr, "wrote %s (%dx%d float32, level=%.3f r=%.3f)\n", out, sz, sz, level, p.grainRadius);
    return 0;
}
