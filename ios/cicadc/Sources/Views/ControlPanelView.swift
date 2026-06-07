import SwiftUI

struct ControlPanelView: View {
    @ObservedObject var vm: ADCViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {

                // ADC Type picker
                VStack(alignment: .leading, spacing: 4) {
                    Label("ADC Type", systemImage: "waveform")
                        .font(.caption.bold())
                        .foregroundColor(.secondary)
                    Picker("ADC Type", selection: $vm.params.adcType) {
                        ForEach(ADCType.allCases) { t in
                            Text(t.rawValue).tag(t)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                Divider()

                // Signal controls
                Group {
                    sliderRow(label: "Frequency",
                              systemImage: "waveform.path",
                              value: $vm.params.frequency,
                              range: 0.1...5.0,
                              format: "%.2f Hz")

                    sliderRow(label: "Amplitude",
                              systemImage: "speaker.wave.2",
                              value: $vm.params.amplitude,
                              range: 0.0...1.0,
                              format: "%.2f FS")

                    sliderRow(label: "Noise",
                              systemImage: "bolt.horizontal",
                              value: $vm.params.noiseAmplitude,
                              range: 0.0...0.5,
                              format: "%.2f FS")

                    sliderRow(label: "Speed",
                              systemImage: "speedometer",
                              value: $vm.params.speed,
                              range: 0.1...4.0,
                              format: "%.1fx")
                }

                Divider()

                // ADC controls
                Group {
                    intSliderRow(label: "Bits",
                                 systemImage: "number",
                                 value: $vm.params.bits,
                                 range: 1...12)

                    sliderRow(label: "Sample Period",
                              systemImage: "clock",
                              value: $vm.params.samplePeriod,
                              range: 0.02...0.5,
                              format: "%.3f s")

                    intSliderRow(label: "Avg Taps",
                                 systemImage: "slider.horizontal.3",
                                 value: $vm.params.avgTaps,
                                 range: 1...64)
                }

                Divider()

                // Dither toggle (only relevant for sigma-delta)
                Toggle(isOn: $vm.params.dither) {
                    Label("Dither", systemImage: "dice")
                }
                .disabled(vm.params.adcType == .nyquist)
                .opacity(vm.params.adcType == .nyquist ? 0.4 : 1)

                Divider()

                // Play / Pause
                Button {
                    vm.togglePlayPause()
                } label: {
                    Label(vm.isPlaying ? "Pause" : "Play",
                          systemImage: vm.isPlaying ? "pause.fill" : "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(vm.isPlaying ? .orange : .green)
            }
            .padding()
        }
    }

    @ViewBuilder
    private func sliderRow(
        label: String, systemImage: String,
        value: Binding<Double>, range: ClosedRange<Double>, format: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Label(label, systemImage: systemImage)
                    .font(.caption.bold())
                    .foregroundColor(.secondary)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .font(.caption.monospacedDigit())
            }
            Slider(value: value, in: range)
                .tint(.accentColor)
        }
    }

    @ViewBuilder
    private func intSliderRow(
        label: String, systemImage: String,
        value: Binding<Int>, range: ClosedRange<Int>
    ) -> some View {
        let doubleBinding = Binding<Double>(
            get: { Double(value.wrappedValue) },
            set: { value.wrappedValue = Int($0.rounded()) }
        )
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Label(label, systemImage: systemImage)
                    .font(.caption.bold())
                    .foregroundColor(.secondary)
                Spacer()
                Text("\(value.wrappedValue)")
                    .font(.caption.monospacedDigit())
            }
            Slider(value: doubleBinding, in: Double(range.lowerBound)...Double(range.upperBound), step: 1)
                .tint(.accentColor)
        }
    }
}
