// extract_frames.swift — pull high-bit-depth frames from a ProRes .mov for grain
// analysis (no ffmpeg required; uses AVFoundation).
//
// Reads RGBA half-float (64bpp) frames, extracts a centered crop, and writes:
//   - a raw binary file of float32 RGBA samples (crop x nFrames), and
//   - a per-frame mean/stddev summary to stdout.
//
// Build:  swiftc -O tools/extract_frames.swift -o extract_frames
// Usage:  ./extract_frames <input.mov> <outDir> <nFrames> <crop> [strideSec]

import Foundation
import AVFoundation
import CoreVideo

func float16to32(_ h: UInt16) -> Float {
    let sign = UInt32(h & 0x8000) << 16
    let exp  = UInt32(h & 0x7C00) >> 10
    let man  = UInt32(h & 0x03FF)
    var bits: UInt32
    if exp == 0 {
        if man == 0 { bits = sign }
        else {
            var e: UInt32 = 0; var m = man
            while (m & 0x0400) == 0 { m <<= 1; e += 1 }
            m &= 0x03FF
            bits = sign | ((127 - 15 - e) << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        bits = sign | 0x7F800000 | (man << 13)
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (man << 13)
    }
    return Float(bitPattern: bits)
}

let args = CommandLine.arguments
guard args.count >= 5 else {
    FileHandle.standardError.write("usage: extract_frames <input.mov> <outDir> <nFrames> <crop> [strideSec]\n".data(using: .utf8)!)
    exit(2)
}
let inPath = args[1]
let outDir = args[2]
let nFrames = Int(args[3]) ?? 16
let crop = Int(args[4]) ?? 512
let strideSec = args.count > 5 ? Double(args[5]) ?? 0 : 0

let url = URL(fileURLWithPath: inPath)
let asset = AVURLAsset(url: url)
let sem = DispatchSemaphore(value: 0)
var track: AVAssetTrack?
var durationSec: Double = 0
asset.loadTracks(withMediaType: .video) { tracks, _ in
    track = tracks?.first
    sem.signal()
}
sem.wait()
guard let vtrack = track else { FileHandle.standardError.write("no video track\n".data(using: .utf8)!); exit(1) }

let durSem = DispatchSemaphore(value: 0)
asset.loadValuesAsynchronously(forKeys: ["duration"]) { durSem.signal() }
durSem.wait()
durationSec = CMTimeGetSeconds(asset.duration)

let dims = vtrack.naturalSize
let W = Int(dims.width), H = Int(dims.height)
FileHandle.standardError.write("video \(W)x\(H) dur=\(durationSec)s, extracting \(nFrames) frames, crop \(crop)\n".data(using: .utf8)!)

try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
let rawPath = "\(outDir)/crop_rgba_f32.bin"
FileManager.default.createFile(atPath: rawPath, contents: nil)
let rawFH = FileHandle(forWritingAtPath: rawPath)!

let cx = max(0, (W - crop) / 2)
let cy = max(0, (H - crop) / 2)
let cw = min(crop, W), ch = min(crop, H)

// Frame times spread across the clip (or fixed stride).
var times: [Double] = []
if strideSec > 0 {
    var t = strideSec
    while t < durationSec && times.count < nFrames { times.append(t); t += strideSec }
} else {
    for i in 0..<nFrames {
        times.append(durationSec * (Double(i) + 0.5) / Double(nFrames))
    }
}

print("frame,timeSec,meanR,meanG,meanB,stdR,stdG,stdB")

let gen = AVAssetImageGenerator(asset: asset)
gen.requestedTimeToleranceBefore = .zero
gen.requestedTimeToleranceAfter = .zero
gen.appliesPreferredTrackTransform = false

// Use an AVAssetReader for accurate decode to 64-bit RGBA half float.
func readFrame(at t: Double) -> [Float]? {
    guard let reader = try? AVAssetReader(asset: asset) else { return nil }
    let outSettings: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_64RGBAHalf
    ]
    let output = AVAssetReaderTrackOutput(track: vtrack, outputSettings: outSettings)
    output.alwaysCopiesSampleData = true
    let start = CMTime(seconds: t, preferredTimescale: 600)
    reader.timeRange = CMTimeRange(start: start, duration: CMTime(seconds: 2, preferredTimescale: 600))
    guard reader.canAdd(output) else { return nil }
    reader.add(output)
    guard reader.startReading() else { return nil }
    guard let sb = output.copyNextSampleBuffer(),
          let pb = CMSampleBufferGetImageBuffer(sb) else { reader.cancelReading(); return nil }
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly); reader.cancelReading() }
    let base = CVPixelBufferGetBaseAddress(pb)!
    let bpr = CVPixelBufferGetBytesPerRow(pb)
    var out = [Float](repeating: 0, count: cw * ch * 4)
    for yy in 0..<ch {
        let row = base.advanced(by: (cy + yy) * bpr).assumingMemoryBound(to: UInt16.self)
        for xx in 0..<cw {
            let p = (cx + xx) * 4
            let o = (yy * cw + xx) * 4
            out[o+0] = float16to32(row[p+0])
            out[o+1] = float16to32(row[p+1])
            out[o+2] = float16to32(row[p+2])
            out[o+3] = float16to32(row[p+3])
        }
    }
    return out
}

var idx = 0
for t in times {
    guard let px = readFrame(at: t) else {
        FileHandle.standardError.write("  skip frame at \(t)s (decode failed)\n".data(using: .utf8)!)
        continue
    }
    // stats
    var s = [Double](repeating: 0, count: 3), s2 = [Double](repeating: 0, count: 3)
    let n = cw * ch
    for i in 0..<n {
        for c in 0..<3 {
            let v = Double(px[i*4+c]); s[c] += v; s2[c] += v*v
        }
    }
    var mean = [Double](repeating: 0, count: 3), sd = [Double](repeating: 0, count: 3)
    for c in 0..<3 { mean[c] = s[c]/Double(n); let v = s2[c]/Double(n) - mean[c]*mean[c]; sd[c] = v>0 ? v.squareRoot() : 0 }
    print(String(format: "%d,%.3f,%.5f,%.5f,%.5f,%.6f,%.6f,%.6f",
                 idx, t, mean[0], mean[1], mean[2], sd[0], sd[1], sd[2]))
    // append raw
    px.withUnsafeBytes { rawFH.write(Data($0)) }
    idx += 1
}
rawFH.closeFile()
FileHandle.standardError.write("wrote \(idx) frames -> \(rawPath) (crop \(cw)x\(ch), RGBA f32)\n".data(using: .utf8)!)
// metadata sidecar
let meta = "\(cw) \(ch) \(idx)\n"
try? meta.write(toFile: "\(outDir)/crop_meta.txt", atomically: true, encoding: .utf8)
