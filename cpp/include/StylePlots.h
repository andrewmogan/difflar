/// helper scripts for plotting in a good style

Int_t kPTRed       = TColor::GetColor(221,61,45);
Int_t kPTOrange    = TColor::GetColor(253,179,102);
Int_t kPTDarkBlue  = TColor::GetColor(74, 123, 183);
Int_t kPTLightBlue = TColor::GetColor(152, 202, 225);
Int_t kPTVibrantBlue = TColor::GetColor(0, 119, 187);
Int_t kPTVibrantCyan = TColor::GetColor(51, 187, 238);
Int_t kPTVibrantMagenta = TColor::GetColor(238,51,119);
Int_t kPTVibrantTeal    = TColor::GetColor(0, 153, 136);

enum DataType {
  kData,
  kSimulation
};

void SetGenericStyle(){

  // Centre title
  gStyle->SetTitleAlign(22);
  gStyle->SetTitleX(.5);
  gStyle->SetTitleY(.95);
  gStyle->SetTitleBorderSize(0);

  // No info box
  gStyle->SetOptStat(0);

  //set the background color to white
  gStyle->SetFillColor(10);
  gStyle->SetFrameFillColor(10);
  gStyle->SetCanvasColor(10);
  gStyle->SetPadColor(10);
  gStyle->SetTitleFillColor(0);
  gStyle->SetStatColor(10);

  // Don't put a colored frame around the plots
  //gStyle->SetFrameBorderMode(0);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);

  // Set the default line color for a fit function to be red
  gStyle->SetFuncColor(kRed);

  // Marker settings
  //  gStyle->SetMarkerStyle(kFullCircle);

  // No border on legends
  gStyle->SetLegendBorderSize(0);

  // Disabled for violating NOvA style guidelines
  // Scientific notation on axes
  //  TGaxis::SetMaxDigits(3);

  // Axis titles
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

  // Prevent ROOT from occasionally automatically zero-suppressing
  gStyle->SetHistMinimumZero();

  // Thicker lines
  gStyle->SetHistLineWidth(2);
  gStyle->SetFrameLineWidth(2);
  gStyle->SetFuncWidth(2);

  // Set the number of tick marks to show
  gStyle->SetNdivisions(506, "xyz");

  // Set the tick mark style
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);

  // Fonts
  const int kGenericFont = 42;
  gStyle->SetStatFont(kGenericFont);
  gStyle->SetLabelFont(kGenericFont, "xyz");
  gStyle->SetTitleFont(kGenericFont, "xyz");
  gStyle->SetTitleFont(kGenericFont, ""); // Apply same setting to plot titles
  gStyle->SetTextFont(kGenericFont);
  gStyle->SetLegendFont(kGenericFont);

}

void ApplyLabel(DataType dt, double xpos = -1, double ypos = -1, Int_t textcol = kBlack){
  std::string ublabel = "MicroBooNE ";
  if (dt ==kData) ublabel+="Data";
  else            ublabel+="Simulation";

  if (xpos == -1) xpos = 0.85;
  if (ypos == -1) ypos = 0.85;

  TLatex* label = new TLatex(xpos, ypos, ublabel.c_str());
  label->SetNDC();
  label->SetTextSize(2/30.);
  label->SetTextAlign(32);
  label->SetTextColor(textcol);
  label->Draw();
}
