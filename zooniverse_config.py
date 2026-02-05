                              

date = "02012026" #this is the download data for the zooniverse data export - all files should be saved with this
ournames = ['masaosako', 'ebellm', 'taceroc', 'fbianco', 'brunosanchez'] # the core users to build ground truth table
p_R = 0.5 # initial probaility of Real: the frequency of real to bogus
chunk_size = 500_000  # Adjust based on your memory
AGREE_THR = 0.7
TEST = False
#TEST = True # should you test on short files or is this the real thing
plotit = True
plotit = False # should you make a bunch of plots
