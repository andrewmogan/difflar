// produces world data and theory summaries for
// - epsilon_l
// - D_L
//
// Theory:
//  - Atrazhev-Timoshkin
//  - BNL parametrization
//
// Datasets:
//  - MicroBooNE
//  - ICARUS

#include "../include/StylePlots.h"

Int_t kMicroBooNEColor = kPTRed;
Int_t kDarkSideColor   = kViolet;
Int_t kICARUSColor     = kBlack;
Int_t kICARUSoColor    = kGray;
Int_t kBNLColor        = kPTDarkBlue;
Int_t kAtrazhevColor   = kPTOrange; //TColor::GetColor(187, 187, 187);
Int_t kBNLParamColor   = kPTVibrantCyan;
bool isDrawMicroBooNE  = true;


//.............................................................................
Double_t bnl_mu_param(double ef){
  float a0 = 551.6;
  float a1 = 7953.7;
  float a2 = 4440.43;
  float a3 = 4.29;
  float a4 = 43.63;
  float a5 = 0.2053;

  ef=ef/1000;

  float num = a0 + a1*ef + a2*std::pow(ef, 1.5) + a3*std::pow(ef, 2.5);
  float den = 1 + (a1/a0)*ef + a4*std::pow(ef, 2) + a5*std::pow(ef,3);

  return (num/den) * std::pow(89./87., -1.5);

}

//.............................................................................
Double_t bnl_el_param(double ef){
  float b0 = 0.0075; 
  float b1 = 742.9;
  float b2 = 3269.6;
  float b3 = 31678.2;
  float T  = 89;
  float T1 = 87;

  ef=ef/1000;

  return ((b0 + b1*ef + b2*ef*ef)   /
         (1+(b1/b0)*ef + b3*ef*ef)) *
         (T/T1);
}

//.............................................................................
// the atrazhev parameterisation comes from running the fit_atrazhev script
Double_t atrazhev_el_param(double ef){

  double offs = 89 * 0.000086173;
  double p0    = -0.000140438;
  double p1    = 1.38788e-05;
  double p2    = 2.19193e-09;
  double p3    = -2.69324e-13;
  double p4    = 1.14616e-17;

  return offs + p0 +
         p1 * ef   +
         p2 * std::pow(ef, 2) +
         p3 * std::pow(ef, 3) +
         p4 * std::pow(ef, 4);

}

//.............................................................................
Double_t bnl_dl_param(double ef){
  return bnl_mu_param(ef) * bnl_el_param(ef);
}

//.............................................................................
Double_t atrazhev_dl_param(double ef){
  return bnl_mu_param(ef) * atrazhev_el_param(ef);
}

//.............................................................................
Double_t einstein_dl_param(double ef){
  double offs = 89 * 0.000086173;
  return offs/(bnl_mu_param(ef)/100000);
}

//.............................................................................
// main
void make_theory_world_data_plots(){

  gStyle->SetOptStat(0);

  gStyle->SetTitleFont(42, "xyz");
  gStyle->SetLabelFont(42, "xyz");
  gStyle->SetTitleSize(.055, "xyz");
  gStyle->SetTitleOffset(.8, "xyz");
  // More space for y-axis to avoid clashing with big numbers
  gStyle->SetTitleOffset(.9, "y");
  // This applies the same settings to the overall plot title
  gStyle->SetTitleSize(.055, "");
  gStyle->SetTitleOffset(.8, "");
  // Axis labels (numbering)
  gStyle->SetLabelSize(.04, "xyz");
  gStyle->SetLabelOffset(.005, "xyz");


  //.............................................
  // datasets
  //.............................................

  // uboone

  Double_t dl_ubx[1]  = {273.9};
  Double_t dl_ubxu[1] = {273.9*1.15-(273.9)};
  Double_t dl_ubxd[1] = {273.9*1.075-(273.9)};
  Double_t dl_uby[1]  = {3.74};
  Double_t dl_ubyu[1]  = {3.74*1.076 - 3.74};
  Double_t dl_ubyd[1]  = {3.74*1.077 - 3.74};

  TGraphAsymmErrors* dl_ub = new TGraphAsymmErrors(1, dl_ubx, dl_uby, dl_ubxd, dl_ubxu, dl_ubyd, dl_ubyu);
  dl_ub->SetMarkerStyle(20);
  dl_ub->SetMarkerColor(kMicroBooNEColor);
  dl_ub->SetLineWidth(2);
  dl_ub->SetLineColor(kMicroBooNEColor);
  dl_ub->SetMarkerSize(0.8);
  
  Double_t el_ubx[1]   = {273.9};
  Double_t el_ubxu[1]  = {273.9*1.15-(273.9)};
  Double_t el_ubxd[1]  = {273.9*1.075-(273.9)};
  Double_t el_uby[1]   = {3.74/bnl_mu_param(273.9)};
  Double_t el_ubyu[1]  = {(3.74/bnl_mu_param(273.9))*1.076 - 3.74/bnl_mu_param(273.9)};
  Double_t el_ubyd[1]  = {(3.74/bnl_mu_param(273.9))*1.077 - 3.74/bnl_mu_param(273.9)};

  TGraphAsymmErrors* el_ub = new TGraphAsymmErrors(1, el_ubx, el_uby, el_ubxd, el_ubxu, el_ubyd, el_ubyu);
  el_ub->SetMarkerStyle(20);
  el_ub->SetMarkerColor(kMicroBooNEColor);
  el_ub->SetLineWidth(2);
  el_ub->SetLineColor(kMicroBooNEColor);

  // darkside
  Double_t dl_darksidex[1]  = {200};
  Double_t dl_darksidexu[1] = {0};
  Double_t dl_darksidexd[1] = {0};
  Double_t dl_darksidey[1]  = {4.12};
  Double_t dl_darksideyu[1]  = {0.09};
  Double_t dl_darksideyd[1]  = {-0.09};
  
  TGraphAsymmErrors* dl_darkside = new TGraphAsymmErrors(1, dl_darksidex, dl_darksidey, dl_darksidexd, dl_darksidexu, dl_darksideyd, dl_darksideyu);
  dl_darkside->SetMarkerStyle(21);
  dl_darkside->SetMarkerColor(kDarkSideColor);
  dl_darkside->SetLineWidth(2);
  dl_darkside->SetLineColor(kDarkSideColor);
  dl_darkside->SetMarkerSize(0.8);

  
  //  TGraphAsymmErrors* el_icarus = new TGraphAsymmErrors(1, el_darksidex, el_darksidey, el_darksidexd, el_darksidexu, el_darksideyd, el_darksideyu);
  // el_darkside->SetMarkerStyle(21);
  // el_darkside->SetMarkerColor(kDarkSideColor);
  // el_darkside->SetLineWidth(2);
  // el_darkside->SetLineColor(kDarkSideColor);

    // icarus old measurements
  Double_t dl_icarusox[1]  = {225};
  Double_t dl_icarusoxu[1] = {350-225};
  Double_t dl_icarusoxd[1] = {225-100};
  Double_t dl_icarusoy[1]  = {4.8};
  Double_t dl_icarusoyu[1]  = {0.2};
  Double_t dl_icarusoyd[1]  = {0.2};

  TGraphAsymmErrors* dl_icaruso = new TGraphAsymmErrors(1, dl_icarusox, dl_icarusoy, dl_icarusoxd, dl_icarusoxu, dl_icarusoyd, dl_icarusoyu);
  dl_icaruso->SetMarkerStyle(21);
  dl_icaruso->SetMarkerColor(kICARUSoColor);
  dl_icaruso->SetLineWidth(2);
  dl_icaruso->SetLineColor(kICARUSoColor);
  dl_icaruso->SetMarkerSize(0.8);

  
  // icarus
  Double_t dl_icarusx[1]  = {493.8};
  Double_t dl_icarusxu[1] = {0};
  Double_t dl_icarusxd[1] = {0};
  Double_t dl_icarusy[1]  = {4.30};
  Double_t dl_icarusyu[1]  = {0.18};
  Double_t dl_icarusyd[1]  = {-0.19};
  
  TGraphAsymmErrors* dl_icarus = new TGraphAsymmErrors(1, dl_icarusx, dl_icarusy, dl_icarusxd, dl_icarusxu, dl_icarusyd, dl_icarusyu);
  dl_icarus->SetMarkerStyle(21);
  dl_icarus->SetMarkerColor(kICARUSColor);
  dl_icarus->SetLineWidth(2);
  dl_icarus->SetLineColor(kICARUSColor);
  dl_icarus->SetMarkerSize(0.8);
  
  Double_t el_icarusx[1]  = {225};
  Double_t el_icarusxu[1] = {350-225};
  Double_t el_icarusxd[1] = {225-100};
  Double_t el_icarusy[1]  = {4.8/bnl_mu_param(225)};
  Double_t el_icarusyu[1]  = {0.2/bnl_mu_param(225)};
  Double_t el_icarusyd[1]  = {0.2/bnl_mu_param(225)};

  TGraphAsymmErrors* el_icarus = new TGraphAsymmErrors(1, el_icarusx, el_icarusy, el_icarusxd, el_icarusxu, el_icarusyd, el_icarusyu);
  el_icarus->SetMarkerStyle(21);
  el_icarus->SetMarkerColor(kICARUSColor);
  el_icarus->SetLineWidth(2);
  el_icarus->SetLineColor(kICARUSColor);

  // bnl
  Double_t el_bnlx[17]   = {0.1*1000     , 0.134*1000   , 0.150*1000   , 0.2*1000     , 0.27*1000    , 0.30*1000    , 
                            0.4*1000     , 0.454*1000   , 0.5*1000     , 0.54*1000    , 0.6*1000     , 0.70*1000    ,
                            0.8*1000     , 1.00*1000    , 1.125*1000   , 1.47*1000   , 2.0*1000};
  Double_t el_bnlxu[17]  = {0            , 0            , 0            , 0            , 0            , 0            , 
                            0            , 0            , 0            , 0            , 0            , 0            ,
                            0            , 0            , 0            , 0            , 0};
  Double_t el_bnlxd[17]  = {0            , 0            , 0            , 0            , 0            , 0            , 
                            0            , 0            , 0            , 0            , 0            , 0            ,
                            0            , 0            , 0            , 0            , 0};
  Double_t el_bnly[17]   = {0.0135       , 0.0212       , 0.0133       , 0.0151       , 0.0222       , 0.0187       , 
                            0.0194       , 0.0200       , 0.0315       , 0.0199       , 0.0271       , 0.0246       ,
                            0.0228       , 0.0331       , 0.0380       , 0.0331       , 0.0099};
  Double_t el_bnlyu[17]  = {0.0153-0.0135, 0.0269-0.0212, 0.0154-0.0133, 0.0167-0.0151, 0.0289-0.0222, 0.0213-0.0187, 
                            0.0218-0.0194, 0.0261-0.0200, 0.0372-0.0315, 0.0338-0.0199, 0.0317-0.0271, 0.0324-0.0246,
                            0.0278-0.0228, 0.0414-0.0331, 0.0503-0.0380, 0.0518-0.0331, 0.0380-0.0099};
  Double_t el_bnlyd[17]  = {0.0135-0.0121, 0.0212-0.0155, 0.0133-0.0113, 0.0151-0.0136, 0.0222-0.0157, 0.0187-0.0163, 
                            0.0194-0.0171, 0.0200-0.0147, 0.0315-0.0257, 0.0199-0.0063, 0.0271-0.0224, 0.0246-0.0173,
                            0.0228-0.0175, 0.0331-0.0250, 0.0380-0.0259, 0.0331-0.0146, 0.0099-0.0050};

  TGraphAsymmErrors* el_bnl = new TGraphAsymmErrors(17, el_bnlx, el_bnly, el_bnlxd, el_bnlxu, el_bnlyd, el_bnlyu);
  el_bnl->SetMarkerStyle(22);
  el_bnl->SetMarkerColor(kBNLColor);
  el_bnl->SetLineWidth(2);
  el_bnl->SetLineColor(kBNLColor);

  Double_t dl_bnly[17]; 
  Double_t dl_bnlyu[17];
  Double_t dl_bnlyd[17];

  for (int i = 0; i < 17; ++i){
    dl_bnly[i] = el_bnly[i]*bnl_mu_param(el_bnlx[i]);
    dl_bnlyu[i] = el_bnlyu[i]*bnl_mu_param(el_bnlx[i]);
    dl_bnlyd[i] = el_bnlyd[i]*bnl_mu_param(el_bnlx[i]);
  }

  TGraphAsymmErrors* dl_bnl = new TGraphAsymmErrors(17, el_bnlx, dl_bnly, el_bnlxd, el_bnlxu, dl_bnlyd, dl_bnlyu);
  dl_bnl->SetMarkerStyle(22);
  dl_bnl->SetMarkerColor(kBNLColor);
  dl_bnl->SetLineWidth(2);
  dl_bnl->SetLineColor(kBNLColor);
  dl_bnl->SetMarkerSize(0.8);



  //.............................................
  // electron energy
  //.............................................
  //SetGenericStyle();

  TCanvas* c1 = new TCanvas("c1", "", 500, 500);
  c1->SetGridy();
  c1->SetGridx();
  c1->SetLogx();
  c1->SetLogy();
  c1->SetLeftMargin(0.14);
  c1->SetBottomMargin(0.12);

  TH2D* el_bg = new TH2D("el_bg", ";E (V/cm);Electron Energy, #epsilon_{L} (eV)", 100, 80, 10000, 100, 5e-3, 2.0e-1);
  el_bg->GetXaxis()->SetTitleOffset(1.05);
  el_bg->GetYaxis()->SetTitleOffset(1.05);
  el_bg->GetXaxis()->CenterTitle();
  el_bg->GetYaxis()->CenterTitle();
  el_bg->Draw();

  TF1* bnl_el = new TF1("bnl_el", "bnl_el_param(x)", 10e-2, 1500);
  bnl_el->SetLineStyle(5);
  bnl_el->SetLineColor(kBNLParamColor);
  bnl_el->SetLineWidth(4);
  bnl_el->Draw("l same");
  
  TF1* atrazhev_el = new TF1("atrazhev_el", "atrazhev_el_param(x)", 10e-2, 10000);
  atrazhev_el->SetLineColor(kAtrazhevColor);
  atrazhev_el->SetLineStyle(kDashed);
  atrazhev_el->SetLineWidth(4);
  atrazhev_el->Draw("l same");
  
  el_icarus->Draw("p same");
  el_bnl->Draw("p same");
  if (isDrawMicroBooNE)
    el_ub->Draw("p same");

  TLegend* leg = new TLegend(0.15, 0.68+(0.2*1./5.), 0.87, 0.88);
  leg->AddEntry(atrazhev_el, "Atrazhev-Timoshkin  ", "l");
  leg->AddEntry(bnl_el     , "Li et al. Parametrization ", "l");
  leg->AddEntry(el_bnl     , "Li et al. Data            ", "pe");
  leg->AddEntry(el_icarus  , "ICARUS              ", "pe");
  leg->AddEntry(dl_icaruso  , "Previous ICARUS Data              ", "pe");
  leg->AddEntry(dl_darkside  , "DarkSide              ", "pe");

  if (isDrawMicroBooNE){
    leg->SetY1(0.68);
    leg->AddEntry(el_ub      , "MicroBooNE Data", "pe");
  }
  leg->Draw("same");
/*
  TLegend* dois = new TLegend(0.47, 0.68+(0.2*1./5.), 0.87, 0.88);
  dois->AddEntry(atrazhev_el, "10.1109/94.689434", "");
  dois->AddEntry(bnl_el     , "10.1016/j.nima.2016.01.094", "");
  dois->AddEntry(el_bnl     , "10.1016/j.nima.2016.01.094", "");
  dois->AddEntry(el_icarus  , "10.1016/0168-9002(94)90996-2", "");
  dois->SetFillStyle(0);
  dois->SetLineWidth(0);
  dois->SetTextColor(kGray+1);
  dois->Draw("same");

  TLatex* names = new TLatex(0.68, 0.91, "A. Lister, A. Mogan");
  names->SetTextColor(kGray+1);
  names->SetNDC();
  names->SetTextSize(1/35.);
  names->SetTextAlign(11);
  names->Draw();
*/

  //.............................................
  // diffusion plots
  //.............................................

  TCanvas* c2 = new TCanvas("c2", "", 500, 500);
  c2->SetGridy();
  c2->SetGridx();
  c2->SetLogx();
  c2->SetLeftMargin(0.14);
  c2->SetBottomMargin(0.12);

  TH2D* dl_bg = new TH2D("dl_bg", ";E (V/cm);D_{L} (cm^{2}/s)", 100, 80, 10000, 100, 0, 18.5);
  dl_bg->GetXaxis()->SetTitleOffset(1.05);
  dl_bg->GetYaxis()->SetTitleOffset(1.05);
  dl_bg->GetXaxis()->CenterTitle();
  dl_bg->GetYaxis()->CenterTitle();
  dl_bg->Draw();

  TF1* bnl_dl = new TF1("bnl_dl", "bnl_dl_param(x)", 10e-2, 1500);
  bnl_dl->SetLineStyle(5);
  bnl_dl->SetLineColor(kBNLParamColor);
  bnl_dl->SetLineWidth(4);
  bnl_dl->Draw("l same");
  
  TF1* atrazhev_dl = new TF1("atrazhev_dl", "atrazhev_dl_param(x)", 10e-2, 10000);
  atrazhev_dl->SetLineColor(kAtrazhevColor);
  atrazhev_dl->SetLineStyle(kDashed);
  atrazhev_dl->SetLineWidth(4);
  atrazhev_dl->Draw("l same");

  /*
  // Experimental thing that never quite worked
  TF1* einstein_dl = new TF1("einstein_dl", "einstein_dl_param(x)", 10e-2, 10000);
  einstein_dl->SetLineColor(kBlack);
  einstein_dl->Draw("l same");
  */

  dl_icarus->Draw("same p");
  dl_icaruso->Draw("same p");
  dl_bnl->Draw("same p");
  dl_darkside->Draw("same p");
  if (isDrawMicroBooNE)
    dl_ub->Draw("same p");
  leg->Draw("same");
  //dois->Draw("same");
  //names->Draw();

  //.............................................
  // Transverse diffusion plot
  //.............................................

  Double_t dt_ubx[1]  = {273.9};
  Double_t dt_ubxu[1] = {273.9*1.15-(273.9)};
  Double_t dt_ubxd[1] = {273.9*1.075-(273.9)};
  Double_t dt_uby[1]  = {3.74};
  Double_t dt_ubyu[1]  = {3.74*1.076 - 3.74};
  Double_t dt_ubyd[1]  = {3.74*1.077 - 3.74};

  TGraphAsymmErrors* dt_ub = new TGraphAsymmErrors(1, dt_ubx, dt_uby, dt_ubxd, dt_ubxu, dt_ubyd, dt_ubyu);
  dt_ub->SetMarkerStyle(20);
  dt_ub->SetMarkerColor(kMicroBooNEColor);
  dt_ub->SetLineWidth(2);
  dt_ub->SetLineColor(kMicroBooNEColor);
  dt_ub->SetMarkerSize(0.8);

  std::string el_name = "el_summary_";
  std::string dl_name = "dl_summary_";

  if (isDrawMicroBooNE){
    el_name+="ub.pdf";
    dl_name+="ub.pdf";
  }
  else {
    el_name+="noub.pdf";
    dl_name+="noub.pdf";
  }

  c1->SaveAs(el_name.c_str());
  c2->SaveAs(dl_name.c_str());
}
