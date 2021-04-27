#!/bin/bash

gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite \
   -sOUTPUTFILE=plot_DM_BRinv_SI.pdf \
   ../Djouadi_mchi_lam.pdf \
   ../Djouadi_mchi_sigmaSI.pdf \
   ../MajoranaDM_mchi_sigmaSI.pdf \
   ../scalarDM_mchi_sigmaSI.pdf \
   ../vectorDM_mchi_sigmaSI.pdf



