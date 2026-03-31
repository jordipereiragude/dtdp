# Repository accompanying "An efficient large neighborhood search method for a delivery territory design problem"

Paper submitted to Computers and Operations Research together with M. Vilà.

## Contents

This repo is divided into several folders

### AlyEtAl

Partial contents of repo https://github.com/ahmed-o-aly/TerritoryDesign detailing code and results from "An efficient probability-based VNS algorithm for delivery territory design" (doi: 10.1016/j.cor.2024.106756).

The following files are included:

* DTDPAlgorithms.py: Base file with code for the VNS approach detailed in the paper
* vns.low.code.py, vns.medium.code.py, vns.high.code.py: Codes used to execute low, medium and high parameter variants of the VNS method
* vns.low.results.txt, vns.medium.results.txt,  vns.high.results.txt: results obtained by these codes. Each line reports the metrics obtained for one instance

### resultsLNS

The folder contains a file with the results from the different LNS variants

* lns.results.txt: results obtained by the proposed LNS method. Each line reports the metrics obtained for one instance

### instances

Folder with instances. The folder contains instances according to description used in "An efficient probability-based VNS algorithm for delivery territory design". The LNS code uses a modified format. To convert files from the original format to the alternative one use accompanying script located in scripts folder

The instances are divided in subfolders as follows:

* group1: planar instances identified by type (planar) number of BUs (500, 600, 700) and instance number (G0 to G9)
* group2: structured instances identified by type (Center, Corners, Diagonal) number of BUs (486, 600, 726) and instance number (G0 to G9)
* group3: structured instances with different attribute distributions identified by type (Center, Corners, Diagonal) number of BUs (486, 600, 726), instance number (G0 to G9) and distribution (l, m or h) per attribute (demand, workers and customers). 

### scripts

Scripts with utilities and analysis of results

* transform.py: converts from graphml (used by VNS) to txt format (used by LNS).


