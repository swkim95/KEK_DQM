#include "TBaux.h"
#include "GuiTypes.h"
#include "TSystem.h"
#include "TStyle.h"
#include <sys/types.h>


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
  fCID_NIM()
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

  if (!fWCPosition)
    return;

  std::vector<std::vector<float>> wcWaves;
  wcWaves.reserve(3);
  wcWaves.push_back(anEvent.GetData(fCID_WCX).pedcorrectedWaveform());
  wcWaves.push_back(anEvent.GetData(fCID_WCY).pedcorrectedWaveform());
  wcWaves.push_back(anEvent.GetData(fCID_NIM).pedcorrectedWaveform());

  auto posVec = GetPosition(wcWaves);
  if (posVec.size() != 2)
    return;

  fWCPosition->Fill(posVec.at(0), posVec.at(1));
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

  fCanvas->cd(1);
  fWCPosition->Draw("colz");

  fCanvas->cd();
  fCanvas->Update();
  if (fDraw) fCanvas->Pad()->Draw();

  TString output = "./output/Run" + std::to_string(fRunNum) + "_AUX.root";
  if (fAuxCut) output = "./output/Run" + std::to_string(fRunNum) + "_AUX_AuxCut.root";
  TFile* outoutFile = new TFile(output, "RECREATE");
  outoutFile->cd();
  fCanvas->Write();
  fWCPosition->Write();
  outoutFile->Close();

  if (fLive) gSystem->ProcessEvents();
  if (!fLive && fApp) fApp->Run(false);

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
