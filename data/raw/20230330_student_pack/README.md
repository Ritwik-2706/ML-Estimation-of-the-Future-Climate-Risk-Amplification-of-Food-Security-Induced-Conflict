# README

The students' project will form part of a larger program of research on using machine learning methods for causal inference in complex multi-scale systems. The application of interest is climate amplified food security induced conflict. One can find further details at the following site.

- https://research.csiro.au/ai4m/causal-inference-in-complex-multiscale-systems/

The remainder of this readme file contains an explanation of files in this directory, and links to some other useful resources.

---

## Directory list

`./docs/`

- contains some relevant documentation and papers

`./docs/climate_commodity_modelling/`

- This paper is the most relevant to your work. Here I built forecast models for vegetable oil commodity prices as a function of climate teleconnection indices representation of El Nino and La Nina. Here we are instead building forecast models for the commodities key to food security, that is: wheat; soy; rice; and maize. You will also look for relationships between these commodities and all of the climate teleconnections I've provided (i.e. not just El Nino and La Nina).
- Here is a seminar that Vassili gave on this topic
  - https://echo360.org.au/media/305c3089-eea6-46ed-8ae7-46dc7021c8d4/public
- Here is a seminar I gave on climate science, modelling and data more broadly to a finance audience
  - https://www.youtube.com/watch?v=PaRTVswDQiM

`./docs/climate_teleconnections/`

- Contains a paper describing the teleconnection data. Only there for your reference no need to go through in detail.

`./docs/jra55/`

- Papers associated with the jra55 dataset. Only there for your reference no need to go through in detail.

`./docs/nnr1/`

- Paper describing the nnr1 dataset. Only there for your reference no need to go through in detail.

`./data/`

- contains the data that we will need in the project

`./data/worldbank `

- monthly prices for various commodities
- https://www.worldbank.org/en/research/commodity-markets
- You will need to convert these to log-returns, then look to forecast the log-returns using information from the climate teleconnections.

`./data/jra55/`

- Contains the climate teleconnection files for the Japanese Reanalysis of the Atmosphere since 1955
- https://jra.kishou.go.jp/JRA-55/index_en.html
- start using the teleconnections in this directory. Time permitting in the project, we could repeat the analysis using the teleconnections in the nnr1 reanalysis below, to determine the sensitivity of the results to the data source.

`./data/nnr1/`

- Contains the climate teleconnection files for the NCEP v1 Reanalysis.
- https://climatedataguide.ucar.edu/climate-data/ncep-ncar-r1-overview

---

# Teleconection dictionary

The following provides a brief explanation of the climate teleconnection is provided. The first column in all of the files contains the date. In most instances, unless otherwise specified, there is one other column containing the teleconnection data itself.

`<jra55 or nnr1>`.AO.csv

- Arctic Oscillation
- https://www.ncei.noaa.gov/access/monitoring/ao/

`<jra55 or nnr1>`.MEI.csv

- Multivariate ENSO (El Nino Southern Oscillation) Index
- https://en.wikipedia.org/wiki/Multivariate_ENSO_index

`<jra55 or nnr1>`.PSA.csv

- Pacific South American
- https://www.nature.com/articles/s43247-021-00295-4
- There are two data columns: PSA1; PSA2.

`<jra55 or nnr1>`.PNA.csv

- Pacific North American
- https://en.wikipedia.org/wiki/Pacific%E2%80%93North_American_teleconnection_pattern

`<jra55 or nnr1>`.IOD.csv

- Indian Ocean Dipole
- https://en.wikipedia.org/wiki/Indian_Ocean_Dipole

`<jra55 or nnr1>`.NHTELE.csv

- Northern Hemisphere related teleconnections
- There are four data columns: NAO+; AR (Atlantic ridge); SCAND (Scandinavian blocking) ; NAO-

`<jra55 or nnr1>`.SAM.csv

- Southern Annual Mode
- http://www.bom.gov.au/climate/sam/

---
