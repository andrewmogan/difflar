void plot_PhiHist1D() {
    TFile *fin = new TFile("WFresults_Plane2_XX_Data.root", "r");
    TH1D *h_phi = (TH1D*)fin->Get("PhiHist1D");

    TCanvas *c = new TCanvas("c", "", 800, 600);

    h_phi->SetLineColor(kAzure+2);
    h_phi->SetLineWidth(2);
    h_phi->GetXaxis()->SetTitle("Track Angle in Wire-Time Plane (degrees)");
    h_phi->GetXaxis()->SetTitleSize(0.045);
    h_phi->GetXaxis()->SetTitleOffset(1.0);
    h_phi->GetYaxis()->SetTitle("Counts");
    h_phi->GetYaxis()->SetTitleSize(0.045);
    h_phi->GetYaxis()->SetTitleOffset(1.0);

    TPaveText *pt = new TPaveText(0.12, 0.65, 0.28, 0.85, "TL NDC");
    pt->SetTextSize(0.048);
    pt->SetTextAlign(13);
    // Remove text box border and fill
    pt->SetBorderSize(0);
    pt->SetFillColor(0);
    pt->AddText("ICARUS Cosmic Data");
    pt->AddText("Run 7033");

    h_phi->Draw();
    pt->Draw("same");
    c->SaveAs("phiHist1D.png", "PNG");
    c->SaveAs("phiHist1D.pdf", "PDF");
}

int main() {
    plot_PhiHist1D();
}
