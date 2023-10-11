#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>
#include <filesystem>

#include "TROOT.h"
#include "ROOT/TThreadedObject.hxx"
#include "TTree.h"
#include "TFile.h"
#include "TSystem.h"
#include "TSystemFile.h"
#include "TString.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TCut.h"
#include "TF1.h"
#include "TFitResult.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TStyle.h"
#include "TList.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"
#include "TVirtualFFT.h"
#include "TFile.h"
#include "TChain.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TSystemDirectory.h"
#include "TStopwatch.h"
#include "TTreeFormula.h"
#include "TLegend.h"
#include "TPaveText.h"

//#include "caloskim/v1/TrackCaloSkimmerObj.h"
#include "../include/caloskim/v2/TrackCaloSkimmerObj.h"

using namespace std;

const float minDistOffset = 13.0; // was 12.0
const float maxDistOffset = 16.0; // was 15.0
//const int timePadding = 50;
//const int timePadding = 100;
const int timePadding = 200; // best for modern samples
const float threshold = 10.0;
//const TF1 *max_nonlin_anode = new TF1("max_nonlin_anode","1.1 - 0.000055*x*x",0,90);
const TF1 *max_nonlin_anode = new TF1("max_nonlin_anode","1.0",0,90);
const TF1 *max_nonlin_cathode = new TF1("max_nonlin_cathode","1.0",0,90);
//const TF1 *max_nonlin_anode = new TF1("max_nonlin_anode","1.1",0,90);
//const TF1 *max_nonlin_cathode = new TF1("max_nonlin_cathode","1.1",0,90);
const int max_offset = 5;

const float angle_min = 20.0;
const float angle_max = 80.0;
const float angle_step = 2.0;

const bool doBaselineCorr = true;
const bool trimExtrema = true;
const float driftVel = 0.1571; // 0.157565 for MC
//const float driftVel = 0.157565;

const float timeTickSF = 0.4;
const float wirePitch = 0.3;
const int maxTicks = 4096;

const float WF_top = 134.96;
const float WF_bottom = -181.86;
const float WF_upstream = -894.951;
const float WF_downstream = 894.951;
const float WF_cathode = 210.215;
const float WF_ACdist = 148.275;

vector<float> TrimVecExtrema(vector<float> inputvec);
void AddFiles(TChain *ch, const char *dir, const char* substr, bool twotrees);
void AddFilesFromList(TChain *ch, const std::string& file_list_path, bool twotrees);

int main(int argc, char** argv)
{
  TStopwatch timer;
  timer.Start();

  gStyle->SetTitleSize(0.065,"T");
  //gErrorIgnoreLevel = kError;
  gErrorIgnoreLevel = kFatal;
  double stops[5] = {0.00,0.34,0.61,0.84,1.00};
  double red[5] = {0.00,0.00,0.87,1.00,0.51};
  double green[5] = {0.00,0.81,1.00,0.20,0.00};
  double blue[5] = {0.51,1.00,0.12,0.00,0.00};
  TColor::CreateGradientColorTable(5,stops,red,green,blue,255);
  gStyle->SetNumberContours(255);
  gStyle->SetOptStat(0);
  TH1::AddDirectory(kFALSE);

  //gSystem->Load("caloskim/v1/TrackCaloSkimmerObj_h.so");
  gSystem->Load("caloskim/v2/TrackCaloSkimmerObj_h.so");
  
  char *dirname;
  char *filetext;
  char *runtype;
  int planenum;
  char *plotname;
  std::string filelist;
  if(argc < 6)
  {
    cout << endl << "Not enough input parameters provided.  Aborting." << endl << endl;
    return -1;
  }
  else
  {
    dirname = (char*) argv[1];
    filetext = (char*) argv[2];
    runtype = (char*) argv[3];
    planenum = atoi(argv[4]);
    plotname = (char*) argv[5];
    filelist = argv[6];
  }

  vector<pair<float, float>> phi_bins((int) (angle_max-angle_min)/angle_step);
  generate(phi_bins.begin(), phi_bins.end(), [] { static float x = angle_min-angle_step; float y = x+2*angle_step; return make_pair(x+=angle_step, y);} );

  TH1F* PhiHist1D = new TH1F("PhiHist1D","",91,-0.5,90.5);
  TH1F* ThetaHist1D = new TH1F("ThetaHist1D","",91,-0.5,90.5);
  
  TH1F* AnodeNonlinearityHist1D = new TH1F("AnodeNonlinearityHist1D","",100,0.0,5.0);
  TH2F* AnodeNonlinearityPhiHist2D = new TH2F("AnodeNonlinearityPhiHist2D","",91,-0.5,90.5,100,0.0,5.0);
  TH1F* CathodeNonlinearityHist1D = new TH1F("CathodeNonlinearityHist1D","",100,0.0,5.0);
  TH2F* CathodeNonlinearityPhiHist2D = new TH2F("CathodeNonlinearityPhiHist2D","",91,-0.5,90.5,100,0.0,5.0);
  
  TH1F* AnodeRecoHist1D[phi_bins.size()];
  TH1F* CathodeRecoHist1D[phi_bins.size()];
  TH2F* AnodeRecoHist2D[phi_bins.size()];
  TH2F* CathodeRecoHist2D[phi_bins.size()];
  TH3F* AnodeRecoHist3D[phi_bins.size()];
  TH3F* CathodeRecoHist3D[phi_bins.size()];

  for(int i = 0; i < phi_bins.size(); i++)
  {
    AnodeRecoHist1D[i] = new TH1F(Form("AnodeRecoHist1D_%.0fto%.0f",phi_bins[i].first,phi_bins[i].second),"",2*timePadding+1,timeTickSF*(-1*timePadding-0.5),timeTickSF*(timePadding+0.5));
    CathodeRecoHist1D[i] = new TH1F(Form("CathodeRecoHist1D_%.0fto%.0f",phi_bins[i].first,phi_bins[i].second),"",2*timePadding+1,timeTickSF*(-1*timePadding-0.5),timeTickSF*(timePadding+0.5));  
    AnodeRecoHist2D[i] = new TH2F(Form("AnodeRecoHist2D_%.0fto%.0f",phi_bins[i].first,phi_bins[i].second),"",2*timePadding+1,timeTickSF*(-1*timePadding-0.5),timeTickSF*(timePadding+0.5),100,0.5,100.5);
    CathodeRecoHist2D[i] = new TH2F(Form("CathodeRecoHist2D_%.0fto%.0f",phi_bins[i].first,phi_bins[i].second),"",2*timePadding+1,timeTickSF*(-1*timePadding-0.5),timeTickSF*(timePadding+0.5),100,0.5,100.5);
    AnodeRecoHist3D[i] = new TH3F(Form("AnodeRecoHist3D_%.0fto%.0f",phi_bins[i].first,phi_bins[i].second),"",11,-5.5,5.5,2*timePadding+1,timeTickSF*(-1*timePadding-0.5),timeTickSF*(timePadding+0.5),100,0.5,100.5);
    CathodeRecoHist3D[i] = new TH3F(Form("CathodeRecoHist3D_%.0fto%.0f",phi_bins[i].first,phi_bins[i].second),"",11,-5.5,5.5,2*timePadding+1,timeTickSF*(-1*timePadding-0.5),timeTickSF*(timePadding+0.5),100,0.5,100.5);
  }
  
  TH2F* AnodeTrackHist2D[phi_bins.size()];
  TH2F* CathodeTrackHist2D[phi_bins.size()];
  TH2F* AnodeTrackUncertHist2D[phi_bins.size()];
  TH2F* CathodeTrackUncertHist2D[phi_bins.size()];
  TH3F* AnodeTrackHist3D[phi_bins.size()];
  TH3F* CathodeTrackHist3D[phi_bins.size()];

  for(int i = 0; i < phi_bins.size(); i++)
  {
    AnodeTrackHist2D[i] = new TH2F(Form("AnodeTrackHist2D_%.0fto%.0f",phi_bins[i].first,phi_bins[i].second),"",11,-5.5,5.5,2*timePadding+1,timeTickSF*(-1*timePadding-0.5),timeTickSF*(timePadding+0.5));
    CathodeTrackHist2D[i] = new TH2F(Form("CathodeTrackHist2D_%.0fto%.0f",phi_bins[i].first,phi_bins[i].second),"",11,-5.5,5.5,2*timePadding+1,timeTickSF*(-1*timePadding-0.5),timeTickSF*(timePadding+0.5));
    AnodeTrackUncertHist2D[i] = new TH2F(Form("AnodeTrackUncertHist2D_%.0fto%.0f",phi_bins[i].first,phi_bins[i].second),"",11,-5.5,5.5,2*timePadding+1,timeTickSF*(-1*timePadding-0.5),timeTickSF*(timePadding+0.5));
    CathodeTrackUncertHist2D[i] = new TH2F(Form("CathodeTrackUncertHist2D_%.0fto%.0f",phi_bins[i].first,phi_bins[i].second),"",11,-5.5,5.5,2*timePadding+1,timeTickSF*(-1*timePadding-0.5),timeTickSF*(timePadding+0.5));
    AnodeTrackHist3D[i] = new TH3F(Form("AnodeTrackHist3D_%.0fto%.0f",phi_bins[i].first,phi_bins[i].second),"",11,-5.5,5.5,2*timePadding+1,timeTickSF*(-1*timePadding-0.5),timeTickSF*(timePadding+0.5),101,-51,151);
    CathodeTrackHist3D[i] = new TH3F(Form("CathodeTrackHist3D_%.0fto%.0f",phi_bins[i].first,phi_bins[i].second),"",11,-5.5,5.5,2*timePadding+1,timeTickSF*(-1*timePadding-0.5),timeTickSF*(timePadding+0.5),101,-51,151);
  }
  
  char* treedirname;
  int cryoNum;
  int tpcNum;
  if(!strcmp(runtype,"XX"))
  {
    treedirname = (char*) "caloskimE/TrackCaloSkim";
    cryoNum = -1;
    tpcNum = -1;
  }
  else if(!strcmp(runtype,"EX"))
  {
    treedirname = (char*) "caloskimE/TrackCaloSkim";
    cryoNum = 0;
    tpcNum = -1;
  }
  else if(!strcmp(runtype,"EE"))
  {
    treedirname = (char*) "caloskimE/TrackCaloSkim";
    cryoNum = 0;
    tpcNum = 0;
  }
  else if(!strcmp(runtype,"EW"))
  {
    treedirname = (char*) "caloskimE/TrackCaloSkim";
    cryoNum = 0;
    tpcNum = 1;
  }
  else if(!strcmp(runtype,"WX"))
  {
    treedirname = (char*) "caloskimW/TrackCaloSkim";
    cryoNum = 1;
    tpcNum = -1;
  }
  else if(!strcmp(runtype,"WE"))
  {
    treedirname = (char*) "caloskimW/TrackCaloSkim";
    cryoNum = 1;
    tpcNum = 0;
  }
  else if(!strcmp(runtype,"WW"))
  {
    treedirname = (char*) "caloskimW/TrackCaloSkim";
    cryoNum = 1;
    tpcNum = 1;
  }

  float wire_dir_ind1[3] = {0.0, 0.0, 1.0};
  float wire_dir_ind2[3] = {0.0, sqrt(3.0)/2.0, 0.5};
  float wire_dir_col[3] = {0.0, sqrt(3.0)/2.0, -0.5};
  if(tpcNum == 1)
  {
    wire_dir_ind2[2] = -0.5;
    wire_dir_col[2] = 0.5;
  }

  TChain* inputfiles = new TChain(treedirname);
  if(cryoNum == -1)
  {
    //AddFiles(inputfiles,dirname,filetext,true);
    AddFilesFromList(inputfiles,filelist,true);
  }
  else
  {
    //AddFiles(inputfiles,dirname,filetext,false);
    AddFilesFromList(inputfiles,filelist,false);
  }

  TTreeReader readerTracks(inputfiles);
  TTreeReaderValue<int> runNum(readerTracks, "trk.meta.run");
  TTreeReaderValue<int> eventNum(readerTracks, "trk.meta.evt");
  TTreeReaderValue<int> tagtype(readerTracks, "selected");
  TTreeReaderValue<int> cryo(readerTracks, "cryostat");
  TTreeReaderValue<float> minTime_TPCE(readerTracks, Form("hit_min_time_p%d_tpcE",planenum));
  TTreeReaderValue<float> maxTime_TPCE(readerTracks, Form("hit_max_time_p%d_tpcE",planenum));
  TTreeReaderValue<float> minTime_TPCW(readerTracks, Form("hit_min_time_p%d_tpcW",planenum));
  TTreeReaderValue<float> maxTime_TPCW(readerTracks, Form("hit_max_time_p%d_tpcW",planenum));
  TTreeReaderArray<float> trackHitTimes(readerTracks, Form("hits%d.h.time",planenum));
  TTreeReaderArray<unsigned short> trackHitWires(readerTracks, Form("hits%d.h.wire",planenum));
  TTreeReaderArray<unsigned short> trackHitTPCs(readerTracks, Form("hits%d.h.tpc",planenum));
  //TTreeReaderArray<float> trackHitXvals(readerTracks, Form("hits%d.h.p.x",planenum));
  //TTreeReaderArray<float> trackHitYvals(readerTracks, Form("hits%d.h.p.y",planenum));
  //TTreeReaderArray<float> trackHitZvals(readerTracks, Form("hits%d.h.p.z",planenum));
  //TTreeReaderArray<float> trackHitXvals(readerTracks, Form("hits%d.h.sp.x",planenum));
  //TTreeReaderArray<float> trackHitYvals(readerTracks, Form("hits%d.h.sp.y",planenum));
  //TTreeReaderArray<float> trackHitZvals(readerTracks, Form("hits%d.h.sp.z",planenum));
  TTreeReaderArray<float> trackHitXvals(readerTracks, Form("hits%d.tp.x",planenum));
  TTreeReaderArray<float> trackHitYvals(readerTracks, Form("hits%d.tp.y",planenum));
  TTreeReaderArray<float> trackHitZvals(readerTracks, Form("hits%d.tp.z",planenum));
  TTreeReaderArray<bool> trackHitIsOnTraj(readerTracks, Form("hits%d.ontraj",planenum));
  TTreeReaderValue<float> trackDirX(readerTracks, "trk.dir.x");
  TTreeReaderValue<float> trackDirY(readerTracks, "trk.dir.y");
  TTreeReaderValue<float> trackDirZ(readerTracks, "trk.dir.z");
  TTreeReaderArray<sbn::WireInfo> trackWf(readerTracks, Form("wires%d",planenum));

  int counter_A[phi_bins.size()] = {0};
  int counter_C[phi_bins.size()] = {0};
  
  vector<float> anodeWFvals[phi_bins.size()][11][2*timePadding+1];
  vector<float> cathodeWFvals[phi_bins.size()][11][2*timePadding+1];

  while(readerTracks.Next())
  {
    if(*tagtype != 1) continue;

    int whichTPC = -1;
    if(*maxTime_TPCE - *minTime_TPCE > *maxTime_TPCW - *minTime_TPCW)
    {
      whichTPC = 0;
    }
    else
    {
      whichTPC = 1;
    }

    if(((cryoNum != -1) && (*cryo != cryoNum)) || ((tpcNum != -1) && (whichTPC != tpcNum))) continue;
  
    float minT_time = 999999;
    float minT_wire;
    float minT_Xval = 999999;
    float minT_Yval;
    float minT_Zval;
    float maxT_time = -999999;
    float maxT_wire;
    float maxT_Xval = -999999;
    float maxT_Yval;
    float maxT_Zval;
    float minW_low = 999999;
    float maxW_low = -999999;
    float minW_high = 999999;
    float maxW_high = -999999;
    float minT_fortimerange = 999999;
    float maxT_fortimerange = -999999;
    for(int k = 0; k < trackHitWires.GetSize(); k++)
    {
      if((trackHitIsOnTraj[k] == true) && (((whichTPC == 0) && (trackHitTPCs[k] < 2)) || ((whichTPC == 1) && (trackHitTPCs[k] >= 2))))
      {
	if((fabs(WF_cathode-fabs(trackHitXvals[k])) > WF_ACdist-maxDistOffset) && (fabs(WF_cathode-fabs(trackHitXvals[k])) < WF_ACdist-minDistOffset) && (fabs(fabs(WF_cathode-fabs(trackHitXvals[k])) - (WF_ACdist-(minDistOffset+maxDistOffset)/2.0)) < fabs(fabs(WF_cathode-fabs(minT_Xval)) - (WF_ACdist-(minDistOffset+maxDistOffset)/2.0))))
	{
          minT_time = trackHitTimes[k];
	  minT_wire = trackHitWires[k];
	  minT_Xval = trackHitXvals[k];
	  minT_Yval = trackHitYvals[k];
	  minT_Zval = trackHitZvals[k];
	}
	if((fabs(WF_cathode-fabs(trackHitXvals[k])) > minDistOffset) && (fabs(WF_cathode-fabs(trackHitXvals[k])) < maxDistOffset) && (fabs(fabs(WF_cathode-fabs(trackHitXvals[k])) - (minDistOffset+maxDistOffset)/2.0) < fabs(fabs(WF_cathode-fabs(maxT_Xval)) - (minDistOffset+maxDistOffset)/2.0)))
	{
          maxT_time = trackHitTimes[k];
	  maxT_wire = trackHitWires[k];
	  maxT_Xval = trackHitXvals[k];
	  maxT_Yval = trackHitYvals[k];
	  maxT_Zval = trackHitZvals[k];
	}

	if(((whichTPC == 0) && (trackHitTPCs[k] == 0)) || ((whichTPC == 1) && (trackHitTPCs[k] == 2)))
	{
          if(trackHitWires[k] < minW_low) minW_low = trackHitWires[k];
          if(trackHitWires[k] > maxW_low) maxW_low = trackHitWires[k];
	}
	if(((whichTPC == 0) && (trackHitTPCs[k] == 1)) || ((whichTPC == 1) && (trackHitTPCs[k] == 3)))
	{
          if(trackHitWires[k] < minW_high) minW_high = trackHitWires[k];
          if(trackHitWires[k] > maxW_high) maxW_high = trackHitWires[k];
	}

        if(trackHitTimes[k] < minT_fortimerange)
	{
	  minT_fortimerange = trackHitTimes[k];
	}
        if(trackHitTimes[k] > maxT_fortimerange)
	{
	  maxT_fortimerange = trackHitTimes[k];
	}
      }
    }

    if((minT_time < 300.0) || (minT_time > maxTicks-300.0)) continue;
    if((maxT_time < 300.0) || (maxT_time > maxTicks-300.0)) continue;

    if((minT_Yval < WF_bottom+10.0) || (minT_Yval > WF_top-10.0)) continue;
    if((maxT_Yval < WF_bottom+10.0) || (maxT_Yval > WF_top-10.0)) continue;

    if((minT_Zval < WF_upstream+10.0) || (minT_Zval > WF_downstream-10.0)) continue;
    if((maxT_Zval < WF_upstream+10.0) || (maxT_Zval > WF_downstream-10.0)) continue;

    if((minT_Zval < 10.0) && (minT_Zval > -10.0)) continue;
    if((maxT_Zval < 10.0) && (maxT_Zval > -10.0)) continue;
    
    float wire_range;
    if((fabs(maxW_low-minW_low) > 10000) && (fabs(maxW_high-minW_high) > 10000))
    {
      continue;
    }
    else if(fabs(maxW_low-minW_low) > 10000)
    {
      wire_range = fabs(maxW_high-minW_high);
    }
    else if(fabs(maxW_high-minW_high) > 10000)
    {
      wire_range = fabs(maxW_low-minW_low);
    }
    else
    {
      wire_range = fabs(maxW_low-minW_low) + fabs(maxW_high-minW_high);
    }

    float track_angle_phi = (180.0/3.14159265)*atan(((maxT_fortimerange-minT_fortimerange)*timeTickSF*driftVel)/(wire_range*wirePitch));
    //if(planenum == 0)
    //{
    //  track_angle_phi = (180.0/3.14159265)*atan((*trackDirX)/((*trackDirY)*wire_dir_ind1[2] - (*trackDirZ)*wire_dir_ind1[1]));
    //}
    //else if(planenum == 1)
    //{
    //  track_angle_phi = (180.0/3.14159265)*atan((*trackDirX)/((*trackDirY)*wire_dir_ind2[2] - (*trackDirZ)*wire_dir_ind2[1]));
    //}
    //else if(planenum == 2)
    //{
    //  track_angle_phi = (180.0/3.14159265)*atan((*trackDirX)/((*trackDirY)*wire_dir_col[2] - (*trackDirZ)*wire_dir_col[1]));
    //}
    //if(track_angle_phi < 0.0)
    //{
    //  track_angle_phi *= -1.0;
    //}
    PhiHist1D->Fill(track_angle_phi);
    //cout << track_angle_phi << " " << (180.0/3.14159265)*atan(((maxT_fortimerange-minT_fortimerange)*timeTickSF*driftVel)/(wire_range*wirePitch)) << endl;

    float track_angle_theta;
    if(planenum == 0)
    {
      track_angle_theta = (180.0/3.14159265)*acos((*trackDirY)*wire_dir_ind1[1] + (*trackDirZ)*wire_dir_ind1[2]);
    }
    else if(planenum == 1)
    {
      track_angle_theta = (180.0/3.14159265)*acos((*trackDirY)*wire_dir_ind2[1] + (*trackDirZ)*wire_dir_ind2[2]);
    }
    else if(planenum == 2)
    {
      track_angle_theta = (180.0/3.14159265)*acos((*trackDirY)*wire_dir_col[1] + (*trackDirZ)*wire_dir_col[2]);
    }
    if(track_angle_theta > 90.0)
    {
      track_angle_theta = 180.0-track_angle_theta;
    }
    ThetaHist1D->Fill(track_angle_theta);

    int phi_index = -1;
    for(int i = 0; i < phi_bins.size(); i++)
    {
      if((track_angle_phi > phi_bins[i].first) && (track_angle_phi < phi_bins[i].second))
      {
        phi_index = i;
      }
    }
    
    if(phi_index < 0)
    {
      continue;
    }

    int anode_minADC = 999999;
    int anode_maxADC = -999999;
    int anode_timeIndex;
    int anode_wireIndex;
    int cathode_minADC = 999999;
    int cathode_maxADC = -999999;
    int cathode_timeIndex;
    int cathode_wireIndex;
    int total_wires = trackWf.GetSize();
    bool anode_flag = false;
    bool cathode_flag = false;
    for(int k = 0; k < total_wires; k++)
    {
      if(((whichTPC == 0) && (trackWf[k].tpc < 2)) || ((whichTPC == 1) && (trackWf[k].tpc >= 2)))
      {
        if(trackWf[k].wire == minT_wire)
        {
    	  anode_flag = true;
    	  
    	  for(int h = 0; h < trackWf[k].adcs.size(); h++)
    	  {
            if(((planenum < 2) && (trackWf[k].adcs[h] < anode_minADC)) || ((planenum == 2) && (trackWf[k].adcs[h] > anode_maxADC)))
    	    {
              anode_timeIndex = h;
    	      anode_wireIndex = k;
    	      if(planenum < 2)
    	      {
                anode_minADC = trackWf[k].adcs[h];
    	      }
    	      else if(planenum == 2)
    	      {
                anode_maxADC = trackWf[k].adcs[h];
    	      }
    	    }
    	  }
        }
        else if(trackWf[k].wire == maxT_wire)
        {
    	  cathode_flag = true;
    	    
    	  for(int h = 0; h < trackWf[k].adcs.size(); h++)
    	  {
            if(((planenum < 2) && (trackWf[k].adcs[h] < cathode_minADC)) || ((planenum == 2) && (trackWf[k].adcs[h] > cathode_maxADC)))
    	    {
              cathode_timeIndex = h;
    	      cathode_wireIndex = k;
    	      if(planenum < 2)
    	      {
                cathode_minADC = trackWf[k].adcs[h];
    	      }
    	      else if(planenum == 2)
    	      {
                cathode_maxADC = trackWf[k].adcs[h];
    	      }
    	    }
    	  }
        }
      }
    }

    if((anode_flag == false) || (cathode_flag == false))
    {
      continue;
    }
    
    int anode_wireIndex_set[11];
    int cathode_wireIndex_set[11];
    
    int wireSF;
    if(trackWf[anode_wireIndex].channel < trackWf[cathode_wireIndex].channel)
    {
      wireSF = 1;
    }
    else if(trackWf[anode_wireIndex].channel > trackWf[cathode_wireIndex].channel)
    {
      wireSF = -1;
    }
    else
    {
      continue;
    }

    if((anode_wireIndex-11 < 0) || (anode_wireIndex+11 >= total_wires) || (cathode_wireIndex-11 < 0) || (cathode_wireIndex+11 >= total_wires))
    {
      continue;
    }

    anode_wireIndex_set[5] = anode_wireIndex;
    cathode_wireIndex_set[5] = cathode_wireIndex;
    int anode_central_channel = trackWf[anode_wireIndex].channel;
    int cathode_central_channel = trackWf[cathode_wireIndex].channel;	

    int anode_count = 1;
    int cathode_count = 1;
    for(int i = 1; i <= 11; i++)
    {
      if((anode_count < 6) && (trackWf[anode_wireIndex+i].channel == anode_central_channel+anode_count))
      {
        anode_wireIndex_set[5+wireSF*anode_count] = anode_wireIndex+i;
        anode_count++;
      }
      if((cathode_count < 6) && (trackWf[cathode_wireIndex+i].channel == cathode_central_channel+cathode_count))
      {
        cathode_wireIndex_set[5+wireSF*cathode_count] = cathode_wireIndex+i;
        cathode_count++;
      }
    }
    if((anode_count < 6) || (cathode_count < 6))
    {
      continue;
    }

    anode_count = 1;
    cathode_count = 1;
    for(int i = 1; i <= 11; i++)
    {
      if((anode_count < 6) && (trackWf[anode_wireIndex-i].channel == anode_central_channel-anode_count))
      {
        anode_wireIndex_set[5-wireSF*anode_count] = anode_wireIndex-i;
        anode_count++;
      }
      if((cathode_count < 6) && (trackWf[cathode_wireIndex-i].channel == cathode_central_channel-cathode_count))
      {
        cathode_wireIndex_set[5-wireSF*cathode_count] = cathode_wireIndex-i;
        cathode_count++;
      }
    }
    if((anode_count < 6) || (cathode_count < 6))
    {
      continue;
    }

    float temp_anodeWFvals[2*max_offset+1][11][2*timePadding+1];
    float temp_cathodeWFvals[2*max_offset+1][11][2*timePadding+1];
  
    float anode_tdc0 = trackWf[anode_wireIndex].tdc0;
    float cathode_tdc0 = trackWf[cathode_wireIndex].tdc0;

    float nonlinearity_A = 99999999.0;
    float nonlinearity_C = 99999999.0;

    int anode_offset_index = max_offset;
    int cathode_offset_index = max_offset;

    for(int i = -1*max_offset; i <= max_offset; i++)
    {
      float deviation_A = 0.0;
      float deviation_C = 0.0;

      float num_A = 0.0;
      float num_C = 0.0;
      
      for(int q = 0; q < 11; q++)
      {
        int neighbor_numADCs = trackWf[anode_wireIndex_set[q]].adcs.size();
        float neighbor_tdc0 = trackWf[anode_wireIndex_set[q]].tdc0;
        float expected_time = timePadding+(((q-5)*wirePitch)/(timeTickSF*driftVel))*tan((3.14159265/180.0)*track_angle_phi);
      	  
        for(int k = 0; k < 2*timePadding+1; k++)
        {
          float val = 0.0;
          if((anode_tdc0-neighbor_tdc0+anode_timeIndex-timePadding+k+i >= 0) && (anode_tdc0-neighbor_tdc0+anode_timeIndex-timePadding+k+i < neighbor_numADCs))
          {
            val = trackWf[anode_wireIndex_set[q]].adcs[anode_tdc0-neighbor_tdc0+anode_timeIndex-timePadding+k+i];
          }
          else if(anode_tdc0-neighbor_tdc0+anode_timeIndex-timePadding+k+i < 0)
          {
            val = trackWf[anode_wireIndex_set[q]].adcs[0];
          }
          else if(anode_tdc0-neighbor_tdc0+anode_timeIndex-timePadding+k+i >= neighbor_numADCs)
          {
            val = trackWf[anode_wireIndex_set[q]].adcs[neighbor_numADCs-1];
          }
      
          //temp_anodeWFvals[i+max_offset][q][k] = val*sin((3.14159265/180.0)*track_angle_theta);
	      temp_anodeWFvals[i+max_offset][q][k] = val;
      
          if(val > threshold)
      	  {
            deviation_A += pow((k-expected_time)*cos((3.14159265/180.0)*track_angle_phi),2);
      	    num_A += 1.0;
      	  }
        }
      }

      for(int q = 0; q < 11; q++)
      {
        int neighbor_numADCs = trackWf[cathode_wireIndex_set[q]].adcs.size();
        float neighbor_tdc0 = trackWf[cathode_wireIndex_set[q]].tdc0;
        float expected_time = timePadding+(((q-5)*wirePitch)/(timeTickSF*driftVel))*tan((3.14159265/180.0)*track_angle_phi);
      	  
        for(int k = 0; k < 2*timePadding+1; k++)
        {
          float val = 0.0;
          if((cathode_tdc0-neighbor_tdc0+cathode_timeIndex-timePadding+k+i >= 0) && (cathode_tdc0-neighbor_tdc0+cathode_timeIndex-timePadding+k+i < neighbor_numADCs))
          {
            val = trackWf[cathode_wireIndex_set[q]].adcs[cathode_tdc0-neighbor_tdc0+cathode_timeIndex-timePadding+k+i];
          }
          else if(cathode_tdc0-neighbor_tdc0+cathode_timeIndex-timePadding+k+i < 0)
          {
            val = trackWf[cathode_wireIndex_set[q]].adcs[0];
          }
          else if(cathode_tdc0-neighbor_tdc0+cathode_timeIndex-timePadding+k+i >= neighbor_numADCs)
          {
            val = trackWf[cathode_wireIndex_set[q]].adcs[neighbor_numADCs-1];
          }
      
          //temp_cathodeWFvals[i+max_offset][q][k] = val*sin((3.14159265/180.0)*track_angle_theta);
	  temp_cathodeWFvals[i+max_offset][q][k] = val;
      
          if(val > threshold)
      	  {
            deviation_C += pow((k-expected_time)*cos((3.14159265/180.0)*track_angle_phi),2);
      	    num_C += 1.0;
      	  }
        }
      }

      if(log10(deviation_A/num_A) < nonlinearity_A)
      {
	nonlinearity_A = log10(deviation_A/num_A);
	anode_offset_index = i+max_offset;
      }

      if(log10(deviation_C/num_C) < nonlinearity_C)
      {
	nonlinearity_C = log10(deviation_C/num_C);
	cathode_offset_index = i+max_offset;
      }
    }
    
    AnodeNonlinearityHist1D->Fill(nonlinearity_A);
    AnodeNonlinearityPhiHist2D->Fill(track_angle_phi,nonlinearity_A);
    CathodeNonlinearityHist1D->Fill(nonlinearity_C);
    CathodeNonlinearityPhiHist2D->Fill(track_angle_phi,nonlinearity_C);
    
    if(nonlinearity_A < max_nonlin_anode->Eval(track_angle_phi))
    {
      for(int q = 0; q < 11; q++)
      {
        for(int k = 0; k < 2*timePadding+1; k++)
        {
          anodeWFvals[phi_index][q][k].push_back(temp_anodeWFvals[anode_offset_index][q][k]);

          if((counter_A[phi_index] < 100) && (q == 5))
          {
            AnodeRecoHist2D[phi_index]->SetBinContent(k+1,counter_A[phi_index]+1,temp_anodeWFvals[anode_offset_index][q][k]);
          }
          if(counter_A[phi_index] < 100)
          {
            AnodeRecoHist3D[phi_index]->SetBinContent(q+1,k+1,counter_A[phi_index]+1,temp_anodeWFvals[anode_offset_index][q][k]);
          }
        }
      }

      counter_A[phi_index]++;
      //cout << "A:  " << counter_A+1 << " " << track_angle_phi << " " << nonlinearity_A << endl;
    }

    if(nonlinearity_C < max_nonlin_cathode->Eval(track_angle_phi))
    {
      for(int q = 0; q < 11; q++)
      {
        for(int k = 0; k < 2*timePadding+1; k++)
	{
          cathodeWFvals[phi_index][q][k].push_back(temp_cathodeWFvals[cathode_offset_index][q][k]);

          if((counter_C[phi_index] < 100) && (q == 5))
          {
            CathodeRecoHist2D[phi_index]->SetBinContent(k+1,counter_C[phi_index]+1,temp_cathodeWFvals[cathode_offset_index][q][k]);
          }
          if(counter_C[phi_index] < 100)
          {
            CathodeRecoHist3D[phi_index]->SetBinContent(q+1,k+1,counter_C[phi_index]+1,temp_cathodeWFvals[cathode_offset_index][q][k]);
          }
	}
      }
      
      counter_C[phi_index]++;
      //cout << "C:  " << counter_C+1 << " " << track_angle_phi << " " << nonlinearity_C << endl;
    }
  }

  int total_tracks_A = 0;
  int total_tracks_C = 0;
  for(int i = 0; i < phi_bins.size(); i++)
  {
    total_tracks_A += anodeWFvals[i][5][0].size();
    total_tracks_C += cathodeWFvals[i][5][0].size();
  }
  
  cout << "Plane " << planenum << " Anode Tracks:  " << total_tracks_A << endl;
  cout << "Plane " << planenum << " Cathode Tracks:  " << total_tracks_C << endl;

  for(int i = 0; i < phi_bins.size(); i++)
  {
    for(int q = 0; q < 11; q++)
    {
      for(int k = 0; k < 2*timePadding+1; k++)
      {
        vector<float> temp_anodeWFvals;
        vector<float> temp_cathodeWFvals;
        if(trimExtrema == true)
        {
          temp_anodeWFvals = TrimVecExtrema(anodeWFvals[i][q][k]);
          temp_cathodeWFvals = TrimVecExtrema(cathodeWFvals[i][q][k]);
        }
        else
        {
          temp_anodeWFvals = anodeWFvals[i][q][k];
          temp_cathodeWFvals = cathodeWFvals[i][q][k];
        }
    	
        float anodeSum = 0.0;
        float cathodeSum = 0.0;
        float anodeSumSq = 0.0;
        float cathodeSumSq = 0.0;
        float anodeN = 0.0;
        float cathodeN = 0.0;
      
        const int numEntriesA = temp_anodeWFvals.size();
        for(int h = 0; h < numEntriesA; h++)
        {
          AnodeTrackHist3D[i]->Fill(q-5,timeTickSF*(k-timePadding),temp_anodeWFvals[h]);
          anodeSum += temp_anodeWFvals[h];
          anodeSumSq += pow(temp_anodeWFvals[h],2);
          anodeN += 1.0;
        }
    
        const int numEntriesC = temp_cathodeWFvals.size();
        for(int h = 0; h < numEntriesC; h++)
        {
          CathodeTrackHist3D[i]->Fill(q-5,timeTickSF*(k-timePadding),temp_cathodeWFvals[h]);
          cathodeSum += temp_cathodeWFvals[h];
          cathodeSumSq += pow(temp_cathodeWFvals[h],2);
          cathodeN += 1.0;
        }
    
        if(anodeN != 0)
        {
          AnodeTrackHist2D[i]->SetBinContent(q+1,k+1,anodeSum/anodeN);
          AnodeTrackUncertHist2D[i]->SetBinContent(q+1,k+1,sqrt(anodeSumSq/anodeN - pow(anodeSum/anodeN,2))/sqrt(anodeN));
        }
        else
        {
          AnodeTrackHist2D[i]->SetBinContent(q+1,k+1,0.0);
          AnodeTrackUncertHist2D[i]->SetBinContent(q+1,k+1,0.0);
        }
    	
        if(cathodeN != 0)
        {
          CathodeTrackHist2D[i]->SetBinContent(q+1,k+1,cathodeSum/cathodeN);
          CathodeTrackUncertHist2D[i]->SetBinContent(q+1,k+1,sqrt(cathodeSumSq/cathodeN - pow(cathodeSum/cathodeN,2))/sqrt(cathodeN));
        }
        else
        {
          CathodeTrackHist2D[i]->SetBinContent(q+1,k+1,0.0);
          CathodeTrackUncertHist2D[i]->SetBinContent(q+1,k+1,0.0);
        }
      }
    }

    for(int k = 0; k < 2*timePadding+1; k++)
    {
      vector<float> temp_anodeWFvals;
      vector<float> temp_cathodeWFvals;
      if(trimExtrema == true)
      {
        temp_anodeWFvals = TrimVecExtrema(anodeWFvals[i][5][k]);
        temp_cathodeWFvals = TrimVecExtrema(cathodeWFvals[i][5][k]);
      }
      else
      {
        temp_anodeWFvals = anodeWFvals[i][5][k];
        temp_cathodeWFvals = cathodeWFvals[i][5][k];
      }
    
      float anodeSum = 0.0;
      float cathodeSum = 0.0; 
      float anodeN = 0.0;
      float cathodeN = 0.0; 
    
      const int numEntriesA = temp_anodeWFvals.size();
      for(int h = 0; h < numEntriesA; h++)
      {
        anodeSum += temp_anodeWFvals[h];
        anodeN += 1.0;
      }
    
      const int numEntriesC = temp_cathodeWFvals.size();
      for(int h = 0; h < numEntriesC; h++)
      {
        cathodeSum += temp_cathodeWFvals[h];
        cathodeN += 1.0;
      }
    
      if(anodeN != 0)
      {
        AnodeRecoHist1D[i]->SetBinContent(k+1,anodeSum/anodeN);
      }
      else
      {
        AnodeRecoHist1D[i]->SetBinContent(k+1,0.0);
      }
    
      if(cathodeN != 0)
      {
        CathodeRecoHist1D[i]->SetBinContent(k+1,cathodeSum/cathodeN);
      }
      else
      {
        CathodeRecoHist1D[i]->SetBinContent(k+1,0.0);
      }
    }
  }

  if(doBaselineCorr == true)
  {
    for(int i = 0; i < phi_bins.size(); i++)
    {
      TH1F *AnodeTrackHist2D_projY = (TH1F*) AnodeTrackHist2D[i]->ProjectionY();
      TH1F *CathodeTrackHist2D_projY = (TH1F*) CathodeTrackHist2D[i]->ProjectionY();
    	  
      float anodeCorr_left_avgTime = 0.0;
      float anodeCorr_left_avgVal = 0.0;
      float anodeCorr_left_N = 0.0;
      float anodeCorr_right_avgTime = 0.0;
      float anodeCorr_right_avgVal = 0.0;
      float anodeCorr_right_N = 0.0;
      	
      float cathodeCorr_left_avgTime = 0.0;
      float cathodeCorr_left_avgVal = 0.0;
      float cathodeCorr_left_N = 0.0;
      float cathodeCorr_right_avgTime = 0.0;
      float cathodeCorr_right_avgVal = 0.0;
      float cathodeCorr_right_N = 0.0;
    
      for(int q = 0; q < 11; q++)
      {
        for(int k = 0; k < 2*timePadding+1; k++)
        {
          if((k < 1*timePadding/5) && (q > 8))
          {
            anodeCorr_left_avgTime += AnodeTrackHist2D_projY->GetBinCenter(k+1);
            anodeCorr_left_avgVal += AnodeTrackHist2D[i]->GetBinContent(q+1,k+1);
            anodeCorr_left_N += 1.0;
    
            cathodeCorr_left_avgTime += CathodeTrackHist2D_projY->GetBinCenter(k+1);
            cathodeCorr_left_avgVal += CathodeTrackHist2D[i]->GetBinContent(q+1,k+1);
            cathodeCorr_left_N += 1.0;
    	  }
          else if((k > 9*timePadding/5) && (q < 2))
          {
            anodeCorr_right_avgTime += AnodeTrackHist2D_projY->GetBinCenter(k+1);
            anodeCorr_right_avgVal += AnodeTrackHist2D[i]->GetBinContent(q+1,k+1);
            anodeCorr_right_N += 1.0;
    	  
            cathodeCorr_right_avgTime += CathodeTrackHist2D_projY->GetBinCenter(k+1);
            cathodeCorr_right_avgVal += CathodeTrackHist2D[i]->GetBinContent(q+1,k+1);
            cathodeCorr_right_N += 1.0;
          }
        }
      }
        
      anodeCorr_left_avgTime /= anodeCorr_left_N;
      anodeCorr_left_avgVal /= anodeCorr_left_N;
      anodeCorr_right_avgTime /= anodeCorr_right_N;
      anodeCorr_right_avgVal /= anodeCorr_right_N;
    
      cathodeCorr_left_avgTime /= cathodeCorr_left_N;
      cathodeCorr_left_avgVal /= cathodeCorr_left_N;
      cathodeCorr_right_avgTime /= cathodeCorr_right_N;
      cathodeCorr_right_avgVal /= cathodeCorr_right_N;
    
      float baselineCorrAnode_slope = (anodeCorr_right_avgVal-anodeCorr_left_avgVal)/(anodeCorr_right_avgTime-anodeCorr_left_avgTime);
      float baselineCorrAnode_intercept = (anodeCorr_left_avgVal+anodeCorr_right_avgVal)/2.0;
    
      float baselineCorrCathode_slope = (cathodeCorr_right_avgVal-cathodeCorr_left_avgVal)/(cathodeCorr_right_avgTime-cathodeCorr_left_avgTime);
      float baselineCorrCathode_intercept = (cathodeCorr_left_avgVal+cathodeCorr_right_avgVal)/2.0;
    
      for(int q = 0; q < 11; q++)
      {
        for(int k = 0; k < 2*timePadding+1; k++)
        {
          AnodeTrackHist2D[i]->SetBinContent(q+1,k+1,AnodeTrackHist2D[i]->GetBinContent(q+1,k+1)-(baselineCorrAnode_slope*AnodeTrackHist2D_projY->GetBinCenter(k+1)+baselineCorrAnode_intercept));
          CathodeTrackHist2D[i]->SetBinContent(q+1,k+1,CathodeTrackHist2D[i]->GetBinContent(q+1,k+1)-(baselineCorrCathode_slope*CathodeTrackHist2D_projY->GetBinCenter(k+1)+baselineCorrCathode_intercept));
        }
      }
    
      anodeCorr_left_avgTime = 0.0;
      anodeCorr_left_avgVal = 0.0;
      anodeCorr_left_N = 0.0;
      anodeCorr_right_avgTime = 0.0;
      anodeCorr_right_avgVal = 0.0;
      anodeCorr_right_N = 0.0;
    
      cathodeCorr_left_avgTime = 0.0;
      cathodeCorr_left_avgVal = 0.0;
      cathodeCorr_left_N = 0.0;
      cathodeCorr_right_avgTime = 0.0;
      cathodeCorr_right_avgVal = 0.0;
      cathodeCorr_right_N = 0.0;
      
      for(int k = 0; k < 2*timePadding+1; k++)
      {
        if(k < 1*timePadding/5)
        {
          anodeCorr_left_avgTime += AnodeRecoHist1D[i]->GetBinCenter(k+1);
          cathodeCorr_left_avgTime += CathodeRecoHist1D[i]->GetBinCenter(k+1);
          anodeCorr_left_avgVal += AnodeRecoHist1D[i]->GetBinContent(k+1);
          cathodeCorr_left_avgVal += CathodeRecoHist1D[i]->GetBinContent(k+1);
          anodeCorr_left_N += 1.0;
          cathodeCorr_left_N += 1.0;
        }
        else if(k > 9*timePadding/5)
        {
          anodeCorr_right_avgTime += AnodeRecoHist1D[i]->GetBinCenter(k+1);
          cathodeCorr_right_avgTime += CathodeRecoHist1D[i]->GetBinCenter(k+1);
          anodeCorr_right_avgVal += AnodeRecoHist1D[i]->GetBinContent(k+1);
          cathodeCorr_right_avgVal += CathodeRecoHist1D[i]->GetBinContent(k+1);
          anodeCorr_right_N += 1.0;
          cathodeCorr_right_N += 1.0;
        }
      }
      anodeCorr_left_avgTime /= anodeCorr_left_N;
      cathodeCorr_left_avgTime /= cathodeCorr_left_N;
      anodeCorr_left_avgVal /= anodeCorr_left_N;
      cathodeCorr_left_avgVal /= cathodeCorr_left_N;
      anodeCorr_right_avgTime /= anodeCorr_right_N;
      cathodeCorr_right_avgTime /= cathodeCorr_right_N;
      anodeCorr_right_avgVal /= anodeCorr_right_N;
      cathodeCorr_right_avgVal /= cathodeCorr_right_N;
      
      baselineCorrAnode_slope = (anodeCorr_right_avgVal-anodeCorr_left_avgVal)/(anodeCorr_right_avgTime-anodeCorr_left_avgTime);
      baselineCorrAnode_intercept = (anodeCorr_left_avgVal+anodeCorr_right_avgVal)/2.0;
      baselineCorrCathode_slope = (cathodeCorr_right_avgVal-cathodeCorr_left_avgVal)/(cathodeCorr_right_avgTime-cathodeCorr_left_avgTime);
      baselineCorrCathode_intercept = (cathodeCorr_left_avgVal+cathodeCorr_right_avgVal)/2.0;
      
      for(int k = 0; k < 2*timePadding+1; k++)
      {
        AnodeRecoHist1D[i]->SetBinContent(k+1,AnodeRecoHist1D[i]->GetBinContent(k+1)-(baselineCorrAnode_slope*AnodeRecoHist1D[i]->GetBinCenter(k+1)+baselineCorrAnode_intercept));
        CathodeRecoHist1D[i]->SetBinContent(k+1,CathodeRecoHist1D[i]->GetBinContent(k+1)-(baselineCorrCathode_slope*CathodeRecoHist1D[i]->GetBinCenter(k+1)+baselineCorrCathode_intercept));
      }
    }
  }
  
  TFile outfile(Form("WFresults_Plane%d_%s.root",planenum,runtype),"RECREATE");
  outfile.cd();

  PhiHist1D->Write();
  ThetaHist1D->Write();

  AnodeNonlinearityHist1D->Write();
  AnodeNonlinearityPhiHist2D->Write();
  CathodeNonlinearityHist1D->Write();
  CathodeNonlinearityPhiHist2D->Write();

  for(int i = 0; i < phi_bins.size(); i++)
  {
    AnodeRecoHist1D[i]->Write();
    CathodeRecoHist1D[i]->Write();
    AnodeRecoHist2D[i]->Write();
    CathodeRecoHist2D[i]->Write();
    AnodeRecoHist3D[i]->Write();
    CathodeRecoHist3D[i]->Write();
    
    AnodeTrackHist2D[i]->Write();
    CathodeTrackHist2D[i]->Write();
    AnodeTrackUncertHist2D[i]->Write();
    CathodeTrackUncertHist2D[i]->Write();
    AnodeTrackHist3D[i]->Write();
    CathodeTrackHist3D[i]->Write();
  }
  
  outfile.Close();

  timer.Stop();
  cout << "WaveformStudy Runtime:  " << timer.RealTime() << " sec." << endl;

  return 0;
}

vector<float> TrimVecExtrema(vector<float> inputvec)
{
  sort(inputvec.begin(), inputvec.end());

  vector<float> newvec;
  for(int i = round(inputvec.size()/10.0); i < round(9.0*inputvec.size()/10.0); i++)
  {
    newvec.push_back(inputvec[i]);
  }
  
  return newvec;
}

void AddFiles(TChain *ch, const char *dir = ".", const char* substr = "", bool twotrees = false)
{
  TSystemDirectory thisdir(dir, dir);
  TList *files = thisdir.GetListOfFiles();

  if(files)
  {
    TSystemFile *file;
    TString fname;
    TIter next(files);
    while((file = (TSystemFile*)next()))
    {
      fname = file->GetName();
      if(!file->IsDirectory() && fname.Contains(substr) && fname.EndsWith(".root"))
      {
	ch->AddFile(Form("%s/%s",(char*)dir,fname.Data()));
	if(twotrees == true)
	{
	  ch->AddFile(Form("%s/%s/caloskimW/TrackCaloSkim",(char*)dir,fname.Data()));
	}
      }
    }
  }

  return;
}
#include <fstream>

#include <fstream>
#include <filesystem>

void AddFilesFromList(TChain* ch, const std::string& file_list_path, bool twotrees = false)
{
    std::ifstream file_list(file_list_path);
    if (!file_list)
    {
        std::cerr << "Error opening file: " << file_list_path << std::endl;
        return;
    }

    std::string file_path;
    while (std::getline(file_list, file_path))
    {
        std::filesystem::path path(file_path);

        if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path) && path.extension() == ".root")
        {
            ch->AddFile(file_path.c_str());
            if (twotrees)
            {
                std::filesystem::path twotrees_path = path / "caloskimW/TrackCaloSkim";
                if (std::filesystem::exists(twotrees_path) && std::filesystem::is_directory(twotrees_path))
                {
                    ch->AddFile(twotrees_path.c_str());
                }
            }
        }
    }

    file_list.close();
}

