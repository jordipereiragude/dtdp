# Repository accompanying "An efficient large neighborhood search method for a delivery territory design problem"

Paper submitted to Computers and Operations Research together with M. Vilà.

## Contents

This repo is divided into several folders

- AlyEtAl

Partial contents of repo https://github.com/ahmed-o-aly/TerritoryDesign detailing code and results from "An efficient probability-based VNS algorithm for delivery territory design" (doi: 10.1016/j.cor.2024.106756).

The following files are included:

* DTDPAlgorithms.py: Base file with code for the VNS approach detailed in the paper
* vns.low.code.py, vns.medium.code.py, vns.high.code.py: Codes used to execute low, medium and high parameter variants of the VNS method
* vns.low.results.txt, vns.medium.results.txt,  vns.high.results.txt: results obtained by these codes. Each line reports the metrics obtained for one instance

- resultsLNS

The folder contains a file with the results from the different LNS variants

* lns.results.txt: results obtained by the proposed LNS method. Each line reports the metrics obtained for one instance


