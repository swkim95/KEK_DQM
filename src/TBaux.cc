#include "TBaux.h"
#include "GuiTypes.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TPaveStats.h"
#include <sys/types.h>


TBaux::TBaux(const YAML::Node fNodePlot_, int fRunNum_, bool fPlotting_, bool fLive_, bool fDraw_, TButility fUtility_)
: fNodeAux(fNodePlot_), fRunNum(fRunNum_), fPlotting(fPlotting_), fLive(fLive_), fDraw(fDraw_), fUtility(fUtility_), fAuxCut(false)
{

  fIsFirst = true;
  fDWCCalib = fNodeAux["DWC"]["CALIB"].as<std::vector<float>>();
  fDWCCenter = fNodeAux["DWC"]["CENTER"].as<std::vector<float>>();
}

void TBaux::init() {

  // fCIDtoPlot.push_back(fUtility.GetCID("CC1"));
  // fCIDtoPlot.push_back(fUtility.GetCID("CC2"));
  fCIDtoPlot.push_back(fUtility.GetCID("DWC1R"));
  fCIDtoPlot.push_back(fUtility.GetCID("DWC1L"));
  fCIDtoPlot.push_back(fUtility.GetCID("DWC1U"));
  fCIDtoPlot.push_back(fUtility.GetCID("DWC1D"));
  fCIDtoPlot.push_back(fUtility.GetCID("DWC2R"));
  fCIDtoPlot.push_back(fUtility.GetCID("DWC2L"));
  fCIDtoPlot.push_back(fUtility.GetCID("DWC2U"));
  fCIDtoPlot.push_back(fUtility.GetCID("DWC2D"));
  // fCIDtoPlot.push_back(fUtility.GetCID("PS"));
  // fCIDtoPlot.push_back(fUtility.GetCID("MC"));
  // fCIDtoPlot.push_back(fUtility.GetCID("TC"));

  // fPScut = fNodeAux["PS"][fMethod].as<double>();
  // fMCcut = fNodeAux["MC"][fMethod].as<double>();
  // fCC1cut = -1;
  // fCC2cut = -1;
  // fPSInitCut = -1;
  // fPSFinCut = -1;
  fDWCPosCut = fNodeAux["DWC"]["POSCUT"].as<double>();
  fDWCCorr = fNodeAux["DWC"]["CORR"].as<double>();

  fDWC1 = new TH2D("DWC1", (TString)"Run " + std::to_string(fRunNum) + " DWC 1 position;X [mm];Y [mm]", 200, -50., 50., 200, -50., 50.);
  fDWC1->SetStats(0);

  fDWC2 = new TH2D("DWC2", "DWC 2 position;X [mm];Y [mm]", 200, -50., 50., 200, -50., 50.);
  fDWC2->SetStats(0);

  fDWCXaxis = new TH2D("DWC_X", "DWC X corr;DWC 1 X [mm];DWC 2 X [mm]", 200, -50., 50., 200, -50., 50.);
  fDWCXaxis->SetStats(0);

  fDWCYaxis = new TH2D("DWC_Y", "DWC Y corr;DWC 1 Y [mm];DWC 2 Y [mm]", 200, -50., 50., 200, -50., 50.);
  fDWCYaxis->SetStats(0);

  // if (fMethod == "IntADC") {
  //   fPS = new TH1D("PS_AUX", ";IntADC;nEvents", 440, -30000., 300000.);
  //   fMC = new TH1D("MC_AUX", ";IntADC;nEvents", 440, -30000., 300000.);
  //   fTC = new TH1D("TC_AUX", ";IntADC;nEvents", 440, -30000., 300000.);
  //   fCC1 = new TH1D("CC1_AUX", ";IntADC;nEvents", 440, -30000., 300000.);
  //   fCC2 = new TH1D("CC2_AUX", ";IntADC;nEvents", 440, -30000., 300000.);
  //   fFrameTop = new TH1D("TopFrame", ";IntADC;nEvents", 440, -30000., 300000.);
  //   fFrameBot = new TH1D("BotFrame", ";IntADC;nEvents", 440, -30000., 300000.);
  // } else if (fMethod == "PeakADC") {
  //   fPS = new TH1D("PS_AUX", ";PeakADC;nEvents", 1152, -512., 4096.);
  //   fMC = new TH1D("MC_AUX", ";PeakADC;nEvents", 1152, -512., 4096.);
  //   fTC = new TH1D("TC_AUX", ";IntADC;nEvents", 1152, -512., 4096.);
  //   fCC1 = new TH1D("CC1_AUX", ";PeakADC;nEvents", 1152, -512., 4096.);
  //   fCC2 = new TH1D("CC2_AUX", ";PeakADC;nEvents", 1152, -512., 4096.);
  //   fFrameTop = new TH1D("TopFrame", ";PeakADC;nEvents", 1152, -512., 4096.);
  //   fFrameBot = new TH1D("BotFrame", ";PeakADC;nEvents", 1152, -512., 4096.);
  // }

  // fPS->SetLineColor(4);
  // fPS->SetLineWidth(2);

  // fMC->SetLineColor(2);
  // fMC->SetLineWidth(2);

  // fTC->SetLineColor(3);
  // fTC->SetLineWidth(2);

  // fCC1->SetLineColor(802);
  // fCC1->SetLineWidth(2);

  // fCC2->SetLineColor(418);
  // fCC2->SetLineWidth(2);

  // fFrameTop->SetStats(0);
  // fFrameBot->SetStats(0);

  fCanvas = new TCanvas("fCanvas_Aux", "fCanvas_Aux", 2000, 1000);
  fCanvas->Divide(2, 2);

  fCanvas->cd(1)->SetRightMargin(0.13);
  fCanvas->cd(2)->SetRightMargin(0.13);
  fCanvas->cd(3)->SetRightMargin(0.13);
  fCanvas->cd(4)->SetRightMargin(0.13);
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

  // fRangeMap.insert(std::make_pair("DWC2R", tConfigNode["DWC2R"].as<std::vector<int>>()));
  // fRangeMap.insert(std::make_pair("PS", tConfigNode["PS"].as<std::vector<int>>()));
  // fRangeMap.insert(std::make_pair("MC", tConfigNode["MC"].as<std::vector<int>>()));
  // fRangeMap.insert(std::make_pair("TC", tConfigNode["TC"].as<std::vector<int>>()));
  // fRangeMap.insert(std::make_pair("CC1", tConfigNode["CC1"].as<std::vector<int>>()));
  // fRangeMap.insert(std::make_pair("CC2", tConfigNode["CC2"].as<std::vector<int>>()));
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

float TBaux::LinearInterp(float x1, float y1, float x2, float y2, float threshold) {
    return x1 + (threshold - y1) * (x2 - x1) / (y2 - y1);
}

float TBaux::GetLeadingEdgeBin(std::vector<float> waveform, float percent) {

  float max = *max_element(waveform.begin() + 1, waveform.begin() + 1001);
  float thr = max * percent;

  for (int i = 1; i < 1000; i++) {
    if (waveform.at(i) < thr && waveform.at(i + 1) > thr) {
          return LinearInterp(static_cast<float>(i), waveform.at(i), static_cast<float>(i + 1), waveform.at(i + 1), thr);
    }
  }
  return -1; // Return -1 if no crossing is found
}

std::vector<float> TBaux::GetPosition(std::vector<std::vector<float>> wave) {

  // 1R 1L 1U 1D 2R 2L 2U 2D
  std::vector<float> tDWCtime = {};
  for (int i = 0; i < 8; i++)
    tDWCtime.push_back(800. * (GetLeadingEdgeBin(wave.at(i), 0.4) / 1000.));


  // // DWC1 horizontal slope, DWC1 horizontal offset, DWC1 vertical slope, DWC1 vertical offset
  // // DWC2 horizontal slope, DWC2 horizontal offset, DWC2 vertical slope, DWC2 vertical offset
  // std::vector<double> fDWCCalib;

  std::vector<float> tDWCpos = {};
  tDWCpos.push_back((tDWCtime.at(0) - tDWCtime.at(1)) * fDWCCalib.at(0) + fDWCCalib.at(1)); // DWC1 X
  tDWCpos.push_back((tDWCtime.at(2) - tDWCtime.at(3)) * fDWCCalib.at(2) + fDWCCalib.at(3)); // DWC1 Y
  tDWCpos.push_back((tDWCtime.at(4) - tDWCtime.at(5)) * fDWCCalib.at(4) + fDWCCalib.at(5)); // DWC2 X
  tDWCpos.push_back((tDWCtime.at(6) - tDWCtime.at(7)) * fDWCCalib.at(6) + fDWCCalib.at(7)); // DWC2 Y

  tDWCpos.at(0) = tDWCpos.at(0) - fDWCCenter.at(0);
  tDWCpos.at(1) = tDWCpos.at(1) - fDWCCenter.at(1);
  tDWCpos.at(2) = tDWCpos.at(2) - fDWCCenter.at(2);
  tDWCpos.at(3) = tDWCpos.at(3) - fDWCCenter.at(3);

  return tDWCpos;
}

void TBaux::Fill(TBevt<TBwaveform> anEvent) {
  
  std::vector<std::vector<float>> tDWCwaves;
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC1R")).pedcorrectedWaveform());
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC1L")).pedcorrectedWaveform());
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC1U")).pedcorrectedWaveform());
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC1D")).pedcorrectedWaveform());
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC2R")).pedcorrectedWaveform());
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC2L")).pedcorrectedWaveform());
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC2U")).pedcorrectedWaveform());
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC2D")).pedcorrectedWaveform());

  auto posVec = GetPosition(tDWCwaves);

  fDWC1->Fill(posVec.at(0), posVec.at(1));
  fDWC2->Fill(posVec.at(2), posVec.at(3));
  fDWCXaxis->Fill(posVec.at(0), posVec.at(2));
  fDWCYaxis->Fill(posVec.at(1), posVec.at(3));

  // fPS->Fill(GetValue(anEvent.GetData(fUtility.GetCID("PS")).waveform(), fRangeMap.at("PS").at(0), fRangeMap.at("PS").at(1)));
  // fMC->Fill(GetValue(anEvent.GetData(fUtility.GetCID("MC")).waveform(), fRangeMap.at("MC").at(0), fRangeMap.at("MC").at(1)));
  // fTC->Fill(GetValue(anEvent.GetData(fUtility.GetCID("TC")).waveform(), fRangeMap.at("TC").at(0), fRangeMap.at("TC").at(1)));
  // fCC1->Fill(GetValue(anEvent.GetData(fUtility.GetCID("CC1")).waveform(), fRangeMap.at("CC1").at(0), fRangeMap.at("CC1").at(1)));
  // fCC2->Fill(GetValue(anEvent.GetData(fUtility.GetCID("CC2")).waveform(), fRangeMap.at("CC2").at(0), fRangeMap.at("CC2").at(1)));
}

bool TBaux::IsPassing(TBevt<TBwaveform> anEvent) {


  std::vector<std::vector<float>> tDWCwaves;
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC1R")).pedcorrectedWaveform());
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC1L")).pedcorrectedWaveform());
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC1U")).pedcorrectedWaveform());
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC1D")).pedcorrectedWaveform());
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC2R")).pedcorrectedWaveform());
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC2L")).pedcorrectedWaveform());
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC2U")).pedcorrectedWaveform());
  tDWCwaves.push_back(anEvent.GetData(fUtility.GetCID("DWC2D")).pedcorrectedWaveform());

  auto posVec = GetPosition(tDWCwaves); // 1X, 1Y, 2X, 2Y

  // if ( !(std::abs(posVec.at(0)) < fDWCPosCut && std::abs(posVec.at(1)) < fDWCPosCut) )
  //   return false;

  // if ( !(std::abs(posVec.at(2)) < fDWCPosCut && std::abs(posVec.at(3)) < fDWCPosCut) )
  //   return false;

  // if ( std::sqrt(posVec.at(0) * posVec.at(0) + posVec.at(1) * posVec.at(1)) > 10 )
  //   return false;

  // if ( std::sqrt(posVec.at(2) * posVec.at(2) + posVec.at(3) * posVec.at(3)) > 10 )
  //   return false;

  if ( std::abs(posVec.at(0) - posVec.at(2)) > fDWCCorr )
    return false;

  if ( std::abs(posVec.at(1) - posVec.at(3)) > fDWCCorr )
    return false;

  // if (fMCcut < GetValue(anEvent.GetData(fUtility.GetCID("MC")).waveform(), fRangeMap.at("MC").at(0), fRangeMap.at("MC").at(1)))
  //   return false;

  //   // - pion: (CC1, CC2) = (pedestal, signal) 선택
  //   // - kaon: (CC1, CC2) = (pedestal, signal) 선택
  //   // - proton: (CC1, CC2) = (안씀, pedestal) 선택
  //   // 셋 다 구현이 되면 좋고, 안되면 pion proton만 되어도 좋습니다. 하나만 되면 pion으로 하고요.
  
  // double tIntADC_PS = GetValue(anEvent.GetData(fUtility.GetCID("PS")).waveform(), fRangeMap.at("PS").at(0), fRangeMap.at("PS").at(1));
  // if (fParticle == "PION") {
  
  //   if ( !(fPSInitCut < tIntADC_PS && tIntADC_PS < fPSFinCut) )
  //     return false;

  //   if (fCC1cut < GetValue(anEvent.GetData(fUtility.GetCID("CC1")).waveform(), fRangeMap.at("CC1").at(0), fRangeMap.at("CC1").at(1)))
  //     return false;

  //   if (fCC2cut > GetValue(anEvent.GetData(fUtility.GetCID("CC2")).waveform(), fRangeMap.at("CC2").at(0), fRangeMap.at("CC2").at(1)))
  //     return false;

  // } else if (fParticle == "KAON") {

  //   if ( !(fPSInitCut < tIntADC_PS && tIntADC_PS < fPSFinCut) )
  //     return false;

  //   if (fCC1cut < GetValue(anEvent.GetData(fUtility.GetCID("CC1")).waveform(), fRangeMap.at("CC1").at(0), fRangeMap.at("CC1").at(1)))
  //     return false;

  //   if (fCC2cut > GetValue(anEvent.GetData(fUtility.GetCID("CC2")).waveform(), fRangeMap.at("CC2").at(0), fRangeMap.at("CC2").at(1)))
  //     return false;

  // } else if (fParticle == "PROTON") {
    
  //   if ( !(fPSInitCut < tIntADC_PS && tIntADC_PS < fPSFinCut) )
  //     return false;

  //   if (fCC2cut < GetValue(anEvent.GetData(fUtility.GetCID("CC2")).waveform(), fRangeMap.at("CC2").at(0), fRangeMap.at("CC2").at(1)))
  //     return false;

  // } else {

  //   if (fPScut > GetValue(anEvent.GetData(fUtility.GetCID("PS")).waveform(), fRangeMap.at("PS").at(0), fRangeMap.at("PS").at(1)))
  //     return false;
  // }

  return true;
}

void TBaux::Draw() {

  fCanvas->cd(1);
  fDWC1->Draw("colz");

  fCanvas->cd(2);
  fDWC2->Draw("colz");

  fCanvas->cd(3);
  fDWCXaxis->Draw("colz");

  fCanvas->cd(4);
  fDWCYaxis->Draw("colz");

  // fCanvas->cd(3);
  // fPS->Draw("Hist");

  // fCanvas->cd(4);
  // fMC->Draw("Hist");

  // fCanvas->cd(7);
  // fTC->Draw("Hist");

  // fCanvas->cd(8);
  // fCC1->Draw("Hist");
  // fCC2->Draw("Hist sames");

  // gSystem->ProcessEvents();
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

  SetMaximum();

  fCanvas->cd(1);
  fDWC1->Draw("colz");

  fCanvas->cd(2);
  fDWC2->Draw("colz");

  fCanvas->cd(3);
  fDWCXaxis->Draw("colz");

  fCanvas->cd(4);
  fDWCYaxis->Draw("colz");

  // fCanvas->cd(3);
  // fPS->Draw("Hist");

  // fCanvas->cd(4);
  // fMC->Draw("Hist");

  // fCanvas->cd(7);
  // fTC->Draw("Hist");

  // fCanvas->cd(8);
  // fCC1->Draw("Hist");
  // fCC2->Draw("Hist sames");


  // fCanvas->cd(8);
  // fFrameBot->Draw("");

  // fCanvas->cd(8);
  // fCC1->Draw("Hist sames");
  // if (fIsFirst) {
  //   fCanvas->Update();
  //   TPaveStats* stat = (TPaveStats*)fCC1->FindObject("stats");
  //   stat->SetTextColor(fCC1->GetLineColor());
  //   stat->SetY2NDC(1.);
  //   stat->SetY1NDC(.8);
  //   stat->SaveStyle();
  // }

  // fCanvas->cd(8);
  // fCC2->Draw("Hist sames");
  // if (fIsFirst) {
  //   fCanvas->Update();
  //   TPaveStats* stat = (TPaveStats*)fCC2->FindObject("stats");
  //   stat->SetTextColor(fCC2->GetLineColor());
  //   stat->SetY2NDC(.8);
  //   stat->SetY1NDC(.6);
  //   stat->SaveStyle();
  // }


  if (fIsFirst) fIsFirst = false;

  // SaveAs("");
  fCanvas->cd();
  fCanvas->Update();
  if (fDraw) fCanvas->Pad()->Draw();

  TString output = "./output/Run" + std::to_string(fRunNum) + "_AUX.root";
  if (fAuxCut) output = "./output/Run" + std::to_string(fRunNum) + "_AUX_AuxCut.root";
  TFile* outoutFile = new TFile(output, "RECREATE");
  outoutFile->cd();
  fCanvas->Write();
  outoutFile->Close();

  // if (!fLive) {
  //   fApp->Run(false);
  // } else {
  //   // gSystem->ProcessEvents();
  //   fApp->Run(true);
  // }
  //

  // std::cout << " before plotting " << std::endl;
  // gSystem->Sleep(1000);
  //
  if (fLive) gSystem->ProcessEvents();
  if (!fLive) fApp->Run(false);

  gSystem->Sleep(1000);
}

void TBaux::SaveAs(TString output) {

  if (output == "")
    output = "./output/Run" + std::to_string(fRunNum) + "_AUX.root";

  TFile* outoutFile = new TFile(output, "RECREATE");
  outoutFile->cd();

  fDWC1->Write();
  fDWC2->Write();
  fDWCXaxis->Write();
  fDWCYaxis->Write();

  // fPS->Write();
  // fMC->Write();
  // fCC1->Write();
  // fCC2->Write();

  outoutFile->Close();
}
