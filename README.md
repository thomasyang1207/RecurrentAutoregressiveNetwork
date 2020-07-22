# Recurrent Autoregressive Networks Implementation #

This repo contains a tensorflow implementation of the object tracker described in Fang et al's 2018 paper, "Recurrent Autoregressive Networks for Online Multi-Object Tracking". Unlike the original RAN model, this implementation only utilizes bounding box motion features (i.e. bounding box displacements). As such, this model is faster and easier to use, at the cost of lower tracking performance.  
