#include "TBaux.h"
#include "GuiTypes.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TBufferJSON.h"
#include <sys/types.h>
#include <fstream>
#include <cstdio>


TBaux::TBaux(const YAML::Node fNodePlot_, int fRunNum_, bool fPlotting_, bool fLive_, bool fDraw_, TButility fUtility_)
: fNodeAux(fNodePlot_),
  fRunNum(fRunNum_),
  fPlotting(fPlotting_),
  fLive(fLive_),
  fDraw(fDraw_),
  fAuxCut(false),
  fUtility(fUtility_),
  fApp(nullptr),
  fCanvas(nullptr),
  fIsFirst(true),
  fMethod(""),
  fWCPosition(nullptr),
  fWCThreshold(0.3),
  fWCCalibration(0.05),
  fWCReference(2, 0.),
  fWCPosCut(-1.),
  fCID_WCX(),
  fCID_WCY(),
  fCID_NIM(),
  fHodoEnabled(false),
  fCID_HodoX(),
  fCID_HodoY(),
  fHodoIntFirst(150),
  fHodoIntLast(350),
  fHodoPeakFirst(150),
  fHodoPeakLast(350),
  fHodoIntADC(nullptr),
  fHodoPeakADC(nullptr),
  fCanvasHodoIntADC(nullptr),
  fCanvasHodoPeakADC(nullptr)
{}

void TBaux::init() {

  const auto nodeWC = fNodeAux["WC"];
  if (!nodeWC) {
    throw std::runtime_error("AUX.WC is not configured in YAML.");
  }

  if (nodeWC["CALIB"]) {
    fWCCalibration = nodeWC["CALIB"].as<double>();
  } else {
    throw std::runtime_error("AUX.WC.CALIB is missing in YAML.");
  }

  if (nodeWC["CENTER"]) {
    fWCReference = nodeWC["CENTER"].as<std::vector<double>>();
    if (fWCReference.size() != 2) {
      throw std::runtime_error("AUX.WC.CENTER must contain exactly two values (X_ref, Y_ref).");
    }
  } else {
    throw std::runtime_error("AUX.WC.CENTER is missing in YAML.");
  }

  if (nodeWC["THRESHOLD"])
    fWCThreshold = nodeWC["THRESHOLD"].as<double>();

  if (nodeWC["POSCUT"])
    fWCPosCut = nodeWC["POSCUT"].as<double>();

  fCIDtoPlot.clear();
  fCID_WCX = fUtility.GetCID("WCX");
  fCID_WCY = fUtility.GetCID("WCY");
  fCID_NIM = fUtility.GetCID("NIM");

  fCIDtoPlot.push_back(fCID_WCX);
  fCIDtoPlot.push_back(fCID_WCY);
  fCIDtoPlot.push_back(fCID_NIM);

  fWCPosition = new TH2D(
    "WC_Position",
    (TString)"Run " + std::to_string(fRunNum) + " Wire Chamber position;X [mm];Y [mm]",
    120, -30., 30., 120, -30., 30.);
  fWCPosition->SetStats(0);

  fCanvas = new TCanvas("fCanvas_WC", "fCanvas_WC", 1200, 800);
  fCanvas->Divide(1, 1);
  fCanvas->cd(1)->SetRightMargin(0.13);

  // ── Hodoscope ─────────────────────────────────────────────────────────
  // 16 X-fibers and 16 Y-fibers. We mirror the dummy naming used in
  // draw_hodoscope.cc:75-107 — the hodoscope mapping isn't ready yet, so
  // both X and Y borrow the existing tower channel names. When the proper
  // mapping arrives, only the two hodoX_names / hodoY_names lists below
  // need to change.
  const std::vector<std::string> hodoX_names = {
    "T1-C","T2-C","T3-C","T4-C","T5-C","T6-C","T7-C","T8-C","T9-C",
    "T1-S","T2-S","T3-S","T4-S","T5-S","T6-S","T7-S"
  };
  const std::vector<std::string> hodoY_names = {
    "T1-C","T2-C","T3-C","T4-C","T5-C","T6-C","T7-C","T8-C","T9-C",
    "T1-S","T2-S","T3-S","T4-S","T5-S","T6-S","T7-S"
  };

  fCID_HodoX.clear();
  fCID_HodoY.clear();
  fCID_HodoX.reserve(hodoX_names.size());
  fCID_HodoY.reserve(hodoY_names.size());
  for (const auto& n : hodoX_names) fCID_HodoX.push_back(fUtility.GetCID(n));
  for (const auto& n : hodoY_names) fCID_HodoY.push_back(fUtility.GetCID(n));

  // Make sure TBread fetches the MIDs hosting these channels too.
  for (const auto& cid : fCID_HodoX) fCIDtoPlot.push_back(cid);
  for (const auto& cid : fCID_HodoY) fCIDtoPlot.push_back(cid);

  fHodoIntADC = new TH2F(
    "hodoscope_intADC",
    (TString)"Run " + std::to_string(fRunNum) + " Hodoscope IntADC;X [fiber];Y [fiber];events",
    16, 0., 16., 16, 0., 16.);
  fHodoIntADC->SetStats(0);

  fHodoPeakADC = new TH2F(
    "hodoscope_peakADC",
    (TString)"Run " + std::to_string(fRunNum) + " Hodoscope PeakADC;X [fiber];Y [fiber];events",
    16, 0., 16., 16, 0., 16.);
  fHodoPeakADC->SetStats(0);

  fCanvasHodoIntADC = new TCanvas("fCanvas_HodoIntADC", "fCanvas_HodoIntADC", 800, 800);
  fCanvasHodoIntADC->cd()->SetRightMargin(0.13);

  fCanvasHodoPeakADC = new TCanvas("fCanvas_HodoPeakADC", "fCanvas_HodoPeakADC", 800, 800);
  fCanvasHodoPeakADC->cd()->SetRightMargin(0.13);

  fHodoEnabled = true;
}

void TBaux::SetParticle(std::string fParticle_) {

  fParticle = fParticle_;

  // if (fParticle == "PION") {
  //   fCC1cut = fNodeAux["PION"]["CC1"].as<double>(); 
  //   fCC2cut = fNodeAux["PION"]["CC2"].as<double>();
  //   fPSInitCut = fNodeAux["PION"]["PS_INIT"].as<double>();
  //   fPSFinCut = fNodeAux["PION"]["PS_FIN"].as<double>();
  // }

  // if (fParticle == "KAON") {
  //   fCC1cut = fNodeAux["KAON"]["CC1"].as<double>();
  //   fCC2cut = fNodeAux["KAON"]["CC2"].as<double>();
  //   fPSInitCut = fNodeAux["KAON"]["PS_INIT"].as<double>();
  //   fPSFinCut = fNodeAux["KAON"]["PS_FIN"].as<double>();
  // }

  // if (fParticle == "PROTON") {
  //   fCC1cut = fNodeAux["PROTON"]["CC1"].as<double>();
  //   fCC2cut = fNodeAux["PROTON"]["CC2"].as<double>();
  //   fPSInitCut = fNodeAux["PROTON"]["PS_INIT"].as<double>();
  //   fPSFinCut = fNodeAux["PROTON"]["PS_FIN"].as<double>();
  // }
}

void TBaux::SetRange(const YAML::Node tConfigNode) {

  // Reserved for future per-channel range configuration (e.g., WCX/WCY windows)
  // Example usage:
  // fRangeMap.insert(std::make_pair("WCX", tConfigNode["WCX"].as<std::vector<int>>()));

  // ModuleConfig.Hodoscope: { INT_RANGE: [a, b], PEAK_RANGE: [c, d] }
  // If the entry is missing, the constructor defaults (150, 350) are kept.
  const auto hodo = tConfigNode["Hodoscope"];
  if (hodo) {
    if (hodo["INT_RANGE"]) {
      const auto r = hodo["INT_RANGE"].as<std::vector<int>>();
      if (r.size() == 2) { fHodoIntFirst = r[0]; fHodoIntLast = r[1]; }
    }
    if (hodo["PEAK_RANGE"]) {
      const auto r = hodo["PEAK_RANGE"].as<std::vector<int>>();
      if (r.size() == 2) { fHodoPeakFirst = r[0]; fHodoPeakLast = r[1]; }
    }
  }
}

double TBaux::GetPeakADC(std::vector<short> waveform, int xInit, int xFin) {
  double ped = 0;
  for (int i = 1; i < 101; i++)
    ped += (double)waveform.at(i) / 100.;

  std::vector<double> pedCorWave;
  for (int i = xInit; i < xFin; i++)
    pedCorWave.push_back(ped - (double)waveform.at(i));

  return *std::max_element(pedCorWave.begin(), pedCorWave.end());
}

double TBaux::GetIntADC(std::vector<short> waveform, int xInit, int xFin) {
  double ped = 0;
  for (int i = 1; i < 101; i++)
    ped += (double)waveform.at(i) / 100.;

  double intADC_ = 0;
  for (int i = xInit; i < xFin; i++)
    intADC_ += ped - (double)waveform.at(i);

  return intADC_;
}

float TBaux::LinearInterp(float x1, float y1, float x2, float y2, float threshold) const {
  return x1 + (threshold - y1) * (x2 - x1) / (y2 - y1);
}

float TBaux::GetLeadingEdgeBin(const std::vector<float>& waveform, float percent) const {

  if (waveform.size() < 1002)
    return -1;

  float max = *std::max_element(waveform.begin() + 1, waveform.begin() + 1001);
  float thr = max * percent;

  for (int i = 1; i < 1000; i++) {
    if (waveform.at(i) < thr && waveform.at(i + 1) > thr) {
      return LinearInterp(static_cast<float>(i), waveform.at(i), static_cast<float>(i + 1), waveform.at(i + 1), thr);
    }
  }
  return -1; // Return -1 if no crossing is found
}

std::vector<float> TBaux::GetPosition(const std::vector<std::vector<float>>& wave) {

  if (wave.size() < 3)
    return {};

  auto binToTime = [](float bin) {
    return 800.f * (bin / 1000.f);
  };

  const float wcxBin = GetLeadingEdgeBin(wave.at(0), static_cast<float>(fWCThreshold));
  const float wcyBin = GetLeadingEdgeBin(wave.at(1), static_cast<float>(fWCThreshold));
  const float nimBin = GetLeadingEdgeBin(wave.at(2), static_cast<float>(fWCThreshold));

  if (wcxBin < 0 || wcyBin < 0 || nimBin < 0)
    return {};

  const float wcxTime = binToTime(wcxBin);
  const float wcyTime = binToTime(wcyBin);
  const float nimTime = binToTime(nimBin);

  const float timeDiffX = nimTime - wcxTime;
  const float timeDiffY = nimTime - wcyTime;

  const float posX = static_cast<float>((fWCReference.at(0) - timeDiffX) * fWCCalibration);
  const float posY = static_cast<float>(-1. * (fWCReference.at(1) - timeDiffY) * fWCCalibration);

  return {posX, posY};
}

void TBaux::Fill(TBevt<TBwaveform> anEvent) {

  // ── Wire chamber position ───────────────────────────────────────────────
  if (fWCPosition) {
    std::vector<std::vector<float>> wcWaves;
    wcWaves.reserve(3);
    wcWaves.push_back(anEvent.GetData(fCID_WCX).pedcorrectedWaveform());
    wcWaves.push_back(anEvent.GetData(fCID_WCY).pedcorrectedWaveform());
    wcWaves.push_back(anEvent.GetData(fCID_NIM).pedcorrectedWaveform());

    const auto posVec = GetPosition(wcWaves);
    if (posVec.size() == 2)
      fWCPosition->Fill(posVec.at(0), posVec.at(1));
  }

  // ── Hodoscope (16x16 IntADC and PeakADC maxima) ─────────────────────────
  if (fHodoEnabled)
    FillHodoscope(anEvent);
}

void TBaux::FillHodoscope(TBevt<TBwaveform> anEvent) {

  if (!fHodoIntADC || !fHodoPeakADC) return;
  if (fCID_HodoX.size() != 16 || fCID_HodoY.size() != 16) return;

  // For each event, find the brightest fiber along X and the brightest
  // fiber along Y (separately for IntADC and PeakADC), then fill the 2D
  // (X,Y) histogram. Mirrors draw_hodoscope.cc::247-266.
  std::vector<float> intADC_X(16, 0.f);
  std::vector<float> intADC_Y(16, 0.f);
  std::vector<float> peakADC_X(16, 0.f);
  std::vector<float> peakADC_Y(16, 0.f);

  // Use TBaux's own GetIntADC / GetPeakADC (same pedestal-corrected
  // integration and max-search as function.h's GetInt / GetPeak, just
  // member functions so we don't have to include function.h — which
  // would create duplicate-symbol linker errors against the standalone
  // draw_*.cc executables that already include it).
  for (int i = 0; i < 16; ++i) {
    const std::vector<short> wfX = anEvent.GetData(fCID_HodoX[i]).waveform();
    const std::vector<short> wfY = anEvent.GetData(fCID_HodoY[i]).waveform();
    if (wfX.size() > static_cast<size_t>(fHodoIntLast)) {
      intADC_X[i]  = static_cast<float>(GetIntADC (wfX, fHodoIntFirst,  fHodoIntLast ));
      peakADC_X[i] = static_cast<float>(GetPeakADC(wfX, fHodoPeakFirst, fHodoPeakLast));
    }
    if (wfY.size() > static_cast<size_t>(fHodoIntLast)) {
      intADC_Y[i]  = static_cast<float>(GetIntADC (wfY, fHodoIntFirst,  fHodoIntLast ));
      peakADC_Y[i] = static_cast<float>(GetPeakADC(wfY, fHodoPeakFirst, fHodoPeakLast));
    }
  }

  const int xIdxInt  = std::max_element(intADC_X.begin(),  intADC_X.end())  - intADC_X.begin();
  const int yIdxInt  = std::max_element(intADC_Y.begin(),  intADC_Y.end())  - intADC_Y.begin();
  const int xIdxPeak = std::max_element(peakADC_X.begin(), peakADC_X.end()) - peakADC_X.begin();
  const int yIdxPeak = std::max_element(peakADC_Y.begin(), peakADC_Y.end()) - peakADC_Y.begin();

  fHodoIntADC ->Fill(xIdxInt  + 0.5f, yIdxInt  + 0.5f, 1);
  fHodoPeakADC->Fill(xIdxPeak + 0.5f, yIdxPeak + 0.5f, 1);
}

bool TBaux::IsPassing(TBevt<TBwaveform> anEvent) {


  std::vector<std::vector<float>> wcWaves;
  wcWaves.reserve(3);
  wcWaves.push_back(anEvent.GetData(fCID_WCX).pedcorrectedWaveform());
  wcWaves.push_back(anEvent.GetData(fCID_WCY).pedcorrectedWaveform());
  wcWaves.push_back(anEvent.GetData(fCID_NIM).pedcorrectedWaveform());

  auto posVec = GetPosition(wcWaves); // X, Y
  if (posVec.size() != 2)
    return false;

  if (fWCPosCut > 0) {
    if (std::abs(posVec.at(0)) > fWCPosCut)
      return false;
    if (std::abs(posVec.at(1)) > fWCPosCut)
      return false;
  }

  return true;
}

void TBaux::Draw() {

  if (!fCanvas || !fWCPosition)
    return;

  fCanvas->cd(1);
  fWCPosition->Draw("colz");

  gSystem->Sleep(1000);
}

void TBaux::SetMaximum() {

  // float max = -999;

  // if (fPS->GetMaximum() > max) max = fPS->GetMaximum();
  // if (fMC->GetMaximum() > max) max = fMC->GetMaximum();
  // if (fTC->GetMaximum() > max) max = fTC->GetMaximum();

  // fFrameTop->GetYaxis()->SetRangeUser(0., max * 1.2);


  // if (fPS->GetMaximum() > fMC->GetMaximum()) fFrameTop->GetYaxis()->SetRangeUser(0., fPS->GetMaximum() * 1.2);
  // else                                       fFrameTop->GetYaxis()->SetRangeUser(0., fMC->GetMaximum() * 1.2);

  // if (fCC1->GetMaximum() > fCC2->GetMaximum()) fFrameBot->GetYaxis()->SetRangeUser(0., fCC1->GetMaximum() * 1.2);
  // else                                         fFrameBot->GetYaxis()->SetRangeUser(0., fCC2->GetMaximum() * 1.2);
}

void TBaux::Update() {

  if (!fCanvas || !fWCPosition)
    return;

  if (fIsFirst) fIsFirst = false;

  SetMaximum();

  // ── Draw all AUX canvases ───────────────────────────────────────────────
  fCanvas->cd(1);
  fWCPosition->Draw("colz");
  fCanvas->cd();
  fCanvas->Update();
  if (fDraw) fCanvas->Pad()->Draw();

  if (fHodoEnabled && fCanvasHodoIntADC && fHodoIntADC) {
    fCanvasHodoIntADC->cd();
    fHodoIntADC->Draw("colz");
    fCanvasHodoIntADC->Update();
    if (fDraw) fCanvasHodoIntADC->Pad()->Draw();
  }

  if (fHodoEnabled && fCanvasHodoPeakADC && fHodoPeakADC) {
    fCanvasHodoPeakADC->cd();
    fHodoPeakADC->Draw("colz");
    fCanvasHodoPeakADC->Update();
    if (fDraw) fCanvasHodoPeakADC->Pad()->Draw();
  }

  // ── Combined AUX ROOT file (WC + hodoscope) ─────────────────────────────
  TString output = "./output/Run" + std::to_string(fRunNum) + "_AUX.root";
  if (fAuxCut) output = "./output/Run" + std::to_string(fRunNum) + "_AUX_AuxCut.root";
  {
    TFile outoutFile(output, "RECREATE");
    outoutFile.cd();
    fCanvas->Write();
    fWCPosition->Write();
    if (fHodoEnabled) {
      if (fCanvasHodoIntADC)  fCanvasHodoIntADC ->Write();
      if (fCanvasHodoPeakADC) fCanvasHodoPeakADC->Write();
      if (fHodoIntADC)        fHodoIntADC ->Write();
      if (fHodoPeakADC)       fHodoPeakADC->Write();
    }
    outoutFile.Close();
  }

  // ── Per-canvas JSON dumps ───────────────────────────────────────────────
  // The web run-browser scans Run<N>_<type>_<method>[_AuxCut]_<canvas>.json.
  // We split the AUX output across three pseudo-"methods" so they show up
  // as distinguishable canvases in the run group:
  //   method=WC         → fCanvas_WC          (wire-chamber position)
  //   method=Hodoscope  → fCanvas_HodoIntADC  (16x16 hit map, IntADC)
  //   method=Hodoscope  → fCanvas_HodoPeakADC (16x16 hit map, PeakADC)
  // Each write is atomic (.tmp -> rename) so a polling LIVE viewer never
  // reads a half-written file.
  auto dumpJSON = [&](TCanvas* canvas, const std::string& methodPart) {
    if (!canvas) return;
    std::string basePrefix = "Run" + std::to_string(fRunNum) + "_AUX_" + methodPart;
    if (fAuxCut) basePrefix += "_AuxCut";
    const std::string canvasName = canvas->GetName();
    const std::string finalPath = "./output/" + basePrefix + "_" + canvasName + ".json";
    const std::string tmpPath = finalPath + ".tmp";
    {
      std::ofstream ofs(tmpPath);
      if (ofs) {
        TString json = TBufferJSON::ToJSON(canvas);
        ofs << json.Data();
      }
    }
    std::rename(tmpPath.c_str(), finalPath.c_str());
  };

  dumpJSON(fCanvas,             "WC");
  if (fHodoEnabled) {
    dumpJSON(fCanvasHodoIntADC,  "Hodoscope");
    dumpJSON(fCanvasHodoPeakADC, "Hodoscope");
  }

  // Process pending GUI events only when the canvas is actually being
  // displayed (--DRAW). In batch mode (web UI / scripted runs) we must
  // NOT call fApp->Run(false): the TApplication is constructed without
  // SetReturnFromRun(true) for non-LIVE, so Run(false) would block here
  // forever — even though all output files have already been written.
  // This mirrors the pattern used in TBplotengine::Update().
  if (fDraw) gSystem->ProcessEvents();

  gSystem->Sleep(1000);
}

void TBaux::SaveAs(TString output) {

  if (output == "")
    output = "./output/Run" + std::to_string(fRunNum) + "_AUX.root";

  if (!fWCPosition)
    return;

  TFile* outoutFile = new TFile(output, "RECREATE");
  outoutFile->cd();

  fWCPosition->Write();

  outoutFile->Close();
}
