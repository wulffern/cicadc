import SwiftUI

struct ContentView: View {
    @StateObject private var vm = ADCViewModel()
    @State private var showControls = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                SignalChainBar(params: vm.params)

                GeometryReader { geo in
                    if geo.size.width > geo.size.height {
                        landscapeLayout(geo: geo)
                    } else {
                        portraitLayout(geo: geo)
                    }
                }
            }
            .navigationTitle("cicadc")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        showControls.toggle()
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                    }
                }
                ToolbarItem(placement: .navigationBarLeading) {
                    Button {
                        vm.togglePlayPause()
                    } label: {
                        Image(systemName: vm.isPlaying ? "pause.fill" : "play.fill")
                    }
                }
            }
            .sheet(isPresented: $showControls) {
                NavigationStack {
                    ControlPanelView(vm: vm)
                        .navigationTitle("Parameters")
                        .navigationBarTitleDisplayMode(.inline)
                        .toolbar {
                            ToolbarItem(placement: .confirmationAction) {
                                Button("Done") { showControls = false }
                            }
                        }
                }
                .presentationDetents([.medium, .large])
            }
        }
        .preferredColorScheme(.dark)
    }

    // Muted grey-green for the coarse modulator/quantizer output — kept dim so the
    // white decimated output reads as the primary digital signal.
    private static let coarseColor = Color(red: 0.46, green: 0.54, blue: 0.46).opacity(0.55)

    /// The analog panel: the (noisy) analog input with the blue car at "now".
    /// When a decimator is active it also carries the grey digital-output car,
    /// lagging by the group delay so the reconstruction latency is visible.
    private func analogView() -> some View {
        SignalView(
            samples: vm.analogSamples,
            tNow: vm.tNow,
            visibleDuration: 4.0,
            color: .green,
            title: "Analog",
            isStaircase: false,
            sampleDots: vm.analogSampleDots,
            trailingCarSamples: vm.hasFilter ? vm.digitalSamples : [],
            trailingCarDelay: vm.groupDelay
        )
    }

    /// The digital panel: a sample-and-hold staircase. When a decimator is active
    /// it shows the coarse output (pale green) plus the filtered output (white).
    private func digitalView() -> some View {
        let primary = vm.hasFilter ? vm.coarseSamples : vm.digitalSamples
        return SignalView(
            samples: primary,
            tNow: vm.tNow,
            visibleDuration: 4.0,
            color: vm.hasFilter ? Self.coarseColor : .cyan,
            title: "Digital",
            isStaircase: true,
            overlaySamples: vm.hasFilter ? vm.digitalSamples : [],
            overlayColor: .white,
            overlayShowsCar: vm.hasFilter,
            sampleDots: primary,
            quantLevels: quantLevels()
        )
    }

    /// Reconstruction levels of the uniform quantizer, drawn as a grid on the
    /// digital panel for Nyquist mode at low bit depths (where they are legible).
    private func quantLevels() -> [Double] {
        guard vm.params.adcType == .nyquist, vm.params.bits <= 5 else { return [] }
        let n = 1 << vm.params.bits
        let step = 2.0 / Double(n)
        return (0..<n).map { -1.0 + (Double($0) + 0.5) * step }
    }

    @ViewBuilder
    private func portraitLayout(geo: GeometryProxy) -> some View {
        let panelH = geo.size.height * 0.35
        let stripH = geo.size.height * 0.17

        VStack(spacing: 4) {
            // Signal panels
            HStack(spacing: 4) {
                analogView()
                digitalView()
            }
            .frame(height: panelH)

            // Analysis strips
            HStack(spacing: 4) {
                ErrorView(
                    samples: vm.errorSamples,
                    tNow: vm.tNow,
                    visibleDuration: 4.0,
                    fullScale: vm.errorFullScale
                )
                FFTView(
                    magnitudes: vm.fftMagnitudes,
                    sampleRate: vm.params.sampleRate
                )
            }
            .frame(height: stripH)

            Spacer(minLength: 0)
        }
        .padding(4)
    }

    @ViewBuilder
    private func landscapeLayout(geo: GeometryProxy) -> some View {
        HStack(spacing: 4) {
            VStack(spacing: 4) {
                analogView()
                ErrorView(
                    samples: vm.errorSamples,
                    tNow: vm.tNow,
                    visibleDuration: 4.0,
                    fullScale: vm.errorFullScale
                )
                .frame(height: geo.size.height * 0.25)
            }
            VStack(spacing: 4) {
                digitalView()
                FFTView(
                    magnitudes: vm.fftMagnitudes,
                    sampleRate: vm.params.sampleRate
                )
                .frame(height: geo.size.height * 0.25)
            }
        }
        .padding(4)
    }
}
