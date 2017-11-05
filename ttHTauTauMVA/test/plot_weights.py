from ROOT import TCanvas, TFile, TTree

def plot_weight(filename, plotname, cuts='1'):
    tcanvas = TCanvas()
    # get trees from file
    file = TFile(filename, 'read')
    
    tree_train = file.Get('TrainTree')
    tree_train.Draw('weight',cuts,'norm')
    h_train = tree_train.GetHistogram()
    h_train.SetLineColor(2)
    #tcanvas.SaveAs(plotname+'_train.png')
    tcanvas.Update()
    
    tree_test = file.Get('TestTree')
    tree_test.Draw('weight',cuts,'norm same')
    h_test = tree_test.GetHistogram()
    h_test.SetLineColor(4)
    #tcanvas.SaveAs(plotname+'_test.png')
    tcanvas.Update()

    tcanvas.SaveAs(plotname+'.png')


plot_weight('/uscms/home/ztao/nobackup/BDTs/tmva/ttV/weighted/tmva_output.root','weights_sig','classID<0.5')
plot_weight('/uscms/home/ztao/nobackup/BDTs/tmva/ttV/weighted/tmva_output.root','weights_bkg','classID>0.5')
plot_weight('/uscms/home/ztao/nobackup/BDTs/tmva/ttV/unweighted/tmva_output.root','unweighted_sig','classID<0.5')
plot_weight('/uscms/home/ztao/nobackup/BDTs/tmva/ttV/unweighted/tmva_output.root','unweighted_bkg','classID>0.5')
plot_weight('/uscms/home/ztao/nobackup/BDTs/tmva/ttV/flipnegweight/tmva_output.root','flipnegweights_sig','classID<0.5')
plot_weight('/uscms/home/ztao/nobackup/BDTs/tmva/ttV/flipnegweight/tmva_output.root','flipnegweights_bkg','classID>0.5')
plot_weight('/uscms/home/ztao/nobackup/BDTs/tmva/ttV/ignorenegweight/tmva_output.root','ignorenegweights_sig','classID<0.5')
plot_weight('/uscms/home/ztao/nobackup/BDTs/tmva/ttV/ignorenegweight/tmva_output.root','ignorenegweights_bkg','classID>0.5')
