#ifndef TBaux_h
#define TBaux_h 1

#include <map>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <chrono>
#include <cmath>
#include <numeric>
#include <functional>

#include "TBconfig.h"
#include "TButility.h"
#include "TBdetector.h"
#include "TBplotengine.h"
#include "TBmid.h"
#include "TBevt.h"

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TLegend.h"

class TBaux
{
public:
  TBaux() = default;
  TBaux(const YAML::Node fNodePlot_, int fRunNum_, bool fPlotting_, bool fLive_, bool fDraw_, TButility fUtility_);
  ~TBaux() {}

  void init();

  void Fill(TBevt<TBwaveform> anEvent);
  void Fill(TBevt<TBfastmode> anEvent) {}

  float LinearInterp(float x1, float y1, float x2, float y2, float threshold);
  float GetLeadingEdgeBin(std::vector<float> waveform, float percent);
  std::vector<float> GetPosition(std::vector<std::vector<float>> wave); // 1R, 1L, 1U, 1D, 2R, 2L, 2U, 2D



  void Draw();
  void Update();

  void SaveAs(TString output = "");

  double GetPeakADC(std::vector<short> waveform, int xInit, int xFin);
  double GetIntADC(std::vector<short> waveform, int xInit, int xFin);

  double GetValue(std::vector<short> waveform, int xInit, int xFin) {

    if(fMethod == "PeakADC")
      return GetPeakADC(waveform, xInit, xFin);

    if(fMethod == "IntADC")
      return GetIntADC(waveform, xInit, xFin);

    return -999;
  }

  std::vector<int> GetUniqueMID() {

    return fUtility.GetUniqueMID(fCIDtoPlot);
  }

  void SetRange(const YAML::Node tConfigNode);
  void SetMethod(std::string fMethod_) { 
    fMethod = fMethod_; 
    if (fMethod == "Overlay" || fMethod == "Avg") fMethod = "IntADC";
  }
  void SetApp(TApplication* fApp_) { fApp = fApp_; }
  void SetAUXCut(bool fAuxCut_) { fAuxCut = fAuxCut_; }
  void SetParticle(std::string fParticle_);

  bool IsPassing(TBevt<TBwaveform> anEvent);

  void SetMaximum();

private:
  const YAML::Node fNodeAux;
  int fRunNum;
  bool fPlotting;
  bool fLive;
  bool fDraw;
  bool fAuxCut;
  std::string fParticle;

  TButility fUtility;

  TApplication* fApp;
  TCanvas* fCanvas;

  bool fIsFirst;

  std::string fMethod;

  TH2D* fDWC1;
  TH2D* fDWC2;
  TH2D* fDWCXaxis;
  TH2D* fDWCYaxis;

  // TH1D* fPS;
  // TH1D* fMC;
  // TH1D* fTC;
  // TH1D* fCC1;
  // TH1D* fCC2;

  // TH1D* fFrameTop;
  // TH1D* fFrameBot;

  // double fPScut;
  // double fPSInitCut;
  // double fPSFinCut;
  // double fMCcut;
  // double fCC1cut;
  // double fCC2cut;
  double fDWCPosCut;
  double fDWCCorr;

  std::vector<TBcid> fCIDtoPlot;
  std::map<std::string, std::vector<int>> fRangeMap;

  // DWC1 horizontal slope, DWC1 horizontal offset, DWC1 vertical slope, DWC1 vertical offset
  // DWC2 horizontal slope, DWC2 horizontal offset, DWC2 vertical slope, DWC2 vertical offset
  std::vector<float> fDWCCalib;

  // DWC_1_X, DWC_1_Y, DWC_2_X, DWC_2_Y
  std::vector<float> fDWCCenter;

};








#endif
