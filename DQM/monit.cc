
#include <iostream>
#include "TBmonit.h"
#include "TBobject.h"
#include "TBsingleWaveform.h"

int main(int argc, char* argv[]) {

  ObjectCollection* obj = new ObjectCollection(argc, argv);
  if (obj->Help())
    return 1;


  std::string aCase;
  obj->GetVariable("type", &aCase);

  std::string aMethod;
  obj->GetVariable("method", &aMethod);

  if (aCase == "single" && aMethod == "Waveform") {
  
    TBsingleWaveform* singleWaveform = new TBsingleWaveform(std::move(obj));
    singleWaveform->Loop();
  } else {
    
    TBmonit<TBwaveform>* monit = new TBmonit<TBwaveform>(std::move(obj));
    monit->Loop();
  }






  return 1;
}
