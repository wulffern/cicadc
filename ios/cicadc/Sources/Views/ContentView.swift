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

    @ViewBuilder
    private func portraitLayout(geo: GeometryProxy) -> some View {
        let panelH = geo.size.height * 0.35
        let stripH = geo.size.height * 0.17

        VStack(spacing: 4) {
            // Signal panels
            HStack(spacing: 4) {
                SignalView(
                    samples: vm.analogSamples,
                    tNow: vm.tNow,
                    visibleDuration: 4.0,
                    color: .green,
                    title: "Analog",
                    isStaircase: false
                )
                SignalView(
                    samples: vm.digitalSamples,
                    tNow: vm.tNow,
                    visibleDuration: 4.0,
                    color: .cyan,
                    title: "Digital",
                    isStaircase: vm.params.adcType == .nyquist
                )
            }
            .frame(height: panelH)

            // Analysis strips
            HStack(spacing: 4) {
                ErrorView(
                    samples: vm.errorSamples,
                    tNow: vm.tNow,
                    visibleDuration: 4.0
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
                SignalView(
                    samples: vm.analogSamples,
                    tNow: vm.tNow,
                    visibleDuration: 4.0,
                    color: .green,
                    title: "Analog",
                    isStaircase: false
                )
                ErrorView(
                    samples: vm.errorSamples,
                    tNow: vm.tNow,
                    visibleDuration: 4.0
                )
                .frame(height: geo.size.height * 0.25)
            }
            VStack(spacing: 4) {
                SignalView(
                    samples: vm.digitalSamples,
                    tNow: vm.tNow,
                    visibleDuration: 4.0,
                    color: .cyan,
                    title: "Digital",
                    isStaircase: vm.params.adcType == .nyquist
                )
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
