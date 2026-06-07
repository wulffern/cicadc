import SwiftUI

/// Renders a scrolling analog or digital signal panel.
struct SignalView: View {
    let samples: [(t: Double, v: Double)]
    let tNow: Double
    let visibleDuration: Double
    let color: Color
    let title: String
    let isStaircase: Bool  // draw sample-and-hold steps (digital) vs smooth curve
    // Optional second trace drawn on top (e.g. the filtered digital output over
    // the coarse modulator output). Always rendered as a staircase, no car.
    var overlaySamples: [(t: Double, v: Double)] = []
    var overlayColor: Color = .white
    var overlayShowsCar = false  // draw the digital-output car on the overlay trace
    // Yellow dots marking the ADC sampling instants.
    var sampleDots: [(t: Double, v: Double)] = []
    // Quantization reconstruction levels to draw as faint horizontal grid lines.
    var quantLevels: [Double] = []
    // An extra (grey) car showing the digital output reconstructed on this trace,
    // lagging "now" by the group delay. Drawn when trailingCarSamples is set.
    var trailingCarSamples: [(t: Double, v: Double)] = []
    var trailingCarDelay: Double = 0

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                // Background grid
                GridOverlay(visibleDuration: visibleDuration, tNow: tNow)

                // Quantization reconstruction levels (faint horizontal lines).
                if !quantLevels.isEmpty {
                    Path { path in
                        for lv in quantLevels {
                            let y = yForV(lv, height: geo.size.height)
                            path.move(to: CGPoint(x: 0, y: y))
                            path.addLine(to: CGPoint(x: geo.size.width, y: y))
                        }
                    }
                    .stroke(color.opacity(0.18), lineWidth: 0.5)
                }

                // Signal path
                if !samples.isEmpty {
                    Path { path in
                        buildPath(path: &path, samples: samples, staircase: isStaircase, size: geo.size)
                    }
                    .stroke(color, lineWidth: 2)

                    // The car rides the "now" point, tilted along the trace tangent.
                    if let nowY = valueAtNow(in: samples, size: geo.size) {
                        CarMarker(
                            angle: tangentAngle(in: samples, size: geo.size),
                            height: 28,
                            tint: color,
                            desaturate: isStaircase
                        )
                        .position(x: xForT(tNow, width: geo.size.width),
                                  y: nowY)
                    }
                }

                // Optional overlay trace (the decimated digital output).
                if !overlaySamples.isEmpty {
                    Path { path in
                        buildPath(path: &path, samples: overlaySamples, staircase: true, size: geo.size)
                    }
                    .stroke(overlayColor, lineWidth: 3)

                    // The digital-output car rides the filtered trace at "now".
                    if overlayShowsCar, let nowY = valueAtNow(in: overlaySamples, size: geo.size) {
                        CarMarker(
                            angle: tangentAngle(in: overlaySamples, size: geo.size),
                            height: 28,
                            tint: overlayColor,
                            desaturate: true
                        )
                        .position(x: xForT(tNow, width: geo.size.width),
                                  y: nowY)
                    }
                }

                // Trailing (grey) digital-output car: the reconstructed output
                // riding this trace, lagging "now" by the group delay.
                if !trailingCarSamples.isEmpty {
                    let tTrail = tNow - trailingCarDelay
                    if let v = interpolatedValue(in: trailingCarSamples, at: tTrail) {
                        CarMarker(
                            angle: tangentAngle(in: trailingCarSamples, size: geo.size, at: tTrail),
                            height: 28,
                            tint: .white,
                            desaturate: true
                        )
                        .position(x: xForT(tTrail, width: geo.size.width),
                                  y: yForV(v, height: geo.size.height))
                    }
                }

                // Yellow sample dots marking the ADC sampling instants.
                if !sampleDots.isEmpty {
                    Path { path in
                        for s in sampleDots {
                            let x = xForT(s.t, width: geo.size.width)
                            guard x >= 0, x <= geo.size.width else { continue }
                            let y = yForV(s.v, height: geo.size.height)
                            path.addEllipse(in: CGRect(x: x - 2, y: y - 2, width: 4, height: 4))
                        }
                    }
                    .fill(Color.yellow)
                }

                // "Now" horizontal line
                Rectangle()
                    .fill(Color.white.opacity(0.25))
                    .frame(height: 1)
                    .offset(y: geo.size.height / 2)

                Text(title)
                    .font(.caption.bold())
                    .foregroundColor(color)
                    .padding(4)
            }
            .background(Color.black)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
    }

    private func xForT(_ t: Double, width: CGFloat) -> CGFloat {
        let tStart = tNow - visibleDuration / 2
        let fraction = (t - tStart) / visibleDuration
        return CGFloat(fraction) * width
    }

    private func yForV(_ v: Double, height: CGFloat) -> CGFloat {
        CGFloat((1 - (v + 1) / 2)) * height
    }

    private func buildPath(path: inout Path, samples: [(t: Double, v: Double)], staircase: Bool, size: CGSize) {
        if staircase {
            buildStaircase(path: &path, samples: samples, size: size)
        } else {
            buildCurve(path: &path, samples: samples, size: size)
        }
    }

    private func buildCurve(path: inout Path, samples: [(t: Double, v: Double)], size: CGSize) {
        var started = false
        for s in samples {
            let x = xForT(s.t, width: size.width)
            let y = yForV(s.v, height: size.height)
            if !started {
                path.move(to: CGPoint(x: x, y: y))
                started = true
            } else {
                path.addLine(to: CGPoint(x: x, y: y))
            }
        }
    }

    private func buildStaircase(path: inout Path, samples: [(t: Double, v: Double)], size: CGSize) {
        var prev: (t: Double, v: Double)?
        for s in samples {
            let x = xForT(s.t, width: size.width)
            let y = yForV(s.v, height: size.height)
            if let p = prev {
                let prevY = yForV(p.v, height: size.height)
                // Horizontal then vertical (sample-and-hold)
                path.addLine(to: CGPoint(x: x, y: prevY))
                path.addLine(to: CGPoint(x: x, y: y))
            } else {
                path.move(to: CGPoint(x: x, y: y))
            }
            prev = s
        }
    }

    private func valueAtNow(in samples: [(t: Double, v: Double)], size: CGSize) -> CGFloat? {
        guard let v = interpolatedValue(in: samples, at: tNow) else { return nil }
        return yForV(v, height: size.height)
    }

    /// Linearly interpolate the trace value at an arbitrary time.
    private func interpolatedValue(in samples: [(t: Double, v: Double)], at t: Double) -> Double? {
        guard samples.count >= 2 else { return nil }
        for i in 1..<samples.count {
            let a = samples[i - 1], b = samples[i]
            if a.t <= t && t <= b.t {
                let frac = (b.t - a.t) > 0 ? (t - a.t) / (b.t - a.t) : 0
                return a.v + (b.v - a.v) * frac
            }
        }
        return nil
    }

    /// Heading so the car (sprite faces "up") points along the trace tangent at
    /// "now", driving toward the future (rightward in time). The slope tilt is
    /// clamped so sample-and-hold jumps don't spin the car wildly.
    private func tangentAngle(in samples: [(t: Double, v: Double)], size: CGSize, at center: Double? = nil) -> Angle {
        let c = center ?? tNow
        let dt = visibleDuration * 0.02
        guard let vBack = interpolatedValue(in: samples, at: c - dt),
              let vFwd = interpolatedValue(in: samples, at: c + dt) else {
            return .degrees(90)  // level, nose pointing right
        }
        let dx = (2 * dt) / visibleDuration * Double(size.width)
        let dy = Double(yForV(vFwd, height: size.height) - yForV(vBack, height: size.height))
        var slope = atan2(dy, dx)
        let limit = Double.pi / 3  // ±60° of tilt
        slope = max(-limit, min(limit, slope))
        return .radians(slope + .pi / 2)
    }
}

struct GridOverlay: View {
    let visibleDuration: Double
    let tNow: Double
    private let gridLines = 5  // horizontal amplitude lines

    var body: some View {
        GeometryReader { geo in
            // Horizontal amplitude grid
            ForEach(0..<gridLines, id: \.self) { i in
                let y = CGFloat(i) / CGFloat(gridLines - 1) * geo.size.height
                Path { path in
                    path.move(to: CGPoint(x: 0, y: y))
                    path.addLine(to: CGPoint(x: geo.size.width, y: y))
                }
                .stroke(Color.gray.opacity(0.3), style: StrokeStyle(lineWidth: 0.5, dash: [4, 4]))
            }
        }
    }
}
